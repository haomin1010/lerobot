# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simulation client for async inference with environments like Libero.

Example command:
```shell
python src/lerobot/async_delta_inference/sim_client.py \
    --env.type=libero \
    --env.task=libero_10 \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --n_episodes=10 \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from lerobot.envs.configs import LiberoEnv  # noqa: F401
from lerobot.envs.factory import make_env
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.io_utils import write_video

from ..async_inference.helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    visualize_action_queue_size,
)
from .configs import SimClientConfig
from .constants import SUPPORTED_ENVS


def _env_obs_to_lerobot_features(env_config) -> dict[str, dict]:
    """Convert environment configuration to lerobot features format.
    
    Converts PolicyFeature objects to hw_features format (like robot.observation_features),
    then uses hw_to_dataset_features to get the proper format.
    """
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.utils.constants import OBS_STR
    
    # Build hw_features dict: {key: shape_tuple or float_type}
    hw_features = {}
    for env_key, policy_feat in env_config.features.items():
        if env_key == "action":
            continue
        
        # Get the mapped key (without "observation." prefix for hw_to_dataset_features)
        mapped_key = env_config.features_map.get(env_key, env_key)
        # Remove "observation." or "observation.state" prefix if present
        if mapped_key.startswith("observation.images."):
            clean_key = mapped_key.replace("observation.images.", "")
        elif mapped_key.startswith("observation.state"):
            clean_key = "state"
        elif mapped_key.startswith("observation."):
            clean_key = mapped_key.replace("observation.", "")
        else:
            clean_key = mapped_key
        
        # Convert PolicyFeature to hw_features format
        if policy_feat.type == FeatureType.VISUAL:
            # Images: use shape tuple
            hw_features[clean_key] = policy_feat.shape
        elif policy_feat.type == FeatureType.STATE:
            # State: use float for each dimension
            # For state features, we need individual joint entries
            for i in range(policy_feat.shape[0]):
                hw_features[f"{clean_key}_{i}"] = float
        else:
            # Other features as float
            hw_features[clean_key] = float
    
    # Use hw_to_dataset_features to create proper format (same as robot_client)
    return hw_to_dataset_features(hw_features, OBS_STR, use_video=False)


def _format_env_observation(obs: dict, env_config, task: str = "") -> RawObservation:
    """Format environment observation to match robot observation format.
    
    Converts Gym environment observation format to the format expected by build_dataset_frame.
    For Libero, this means:
    - 'pixels/image' -> 'image' (build_dataset_frame will add prefix)
    - 'pixels/image2' -> 'image2'
    - 'agent_pos' array -> individual 'state_0', 'state_1', ... keys
    """
    from lerobot.configs.types import FeatureType
    
    raw_obs: RawObservation = {}
    
    # Process images - use keys WITHOUT "observation.images." prefix
    # because build_dataset_frame will add it
    if "pixels" in obs:
        for cam_name, img in obs["pixels"].items():
            # Map camera names using features_map from env_config
            full_key = env_config.features_map.get(f"pixels/{cam_name}", f"{OBS_IMAGES}.{cam_name}")
            
            # Remove "observation.images." prefix for build_dataset_frame
            if full_key.startswith("observation.images."):
                clean_key = full_key.replace("observation.images.", "")
            else:
                clean_key = full_key
            
            # Convert numpy array to torch tensor if needed
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            raw_obs[clean_key] = img
    
    # Process state - expand array into individual keys
    if "agent_pos" in obs:
        state = obs["agent_pos"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
        
        # Expand state into individual components (like robot joints)
        # This matches what hw_to_dataset_features expects
        for i in range(len(state)):
            raw_obs[f"state_{i}"] = state[i]
    
    # Always add task (even if empty string) - some policies (like VLAs) may require it
    raw_obs["task"] = task if task is not None else ""
    
    return raw_obs


class SimClient:
    """Simulation client for async inference.
    
    This client is similar to RobotClient but works with simulation environments
    like Libero instead of real robots.
    """
    
    prefix = "sim_client"
    logger = get_logger(prefix)

    def __init__(self, config: SimClientConfig):
        """Initialize SimClient with unified configuration.

        Args:
            config: SimClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        
        # Create the simulation environment
        self.logger.info(f"Creating simulation environment: {config.env.type}")
        env_dict = make_env(config.env, n_envs=1, use_async_envs=False)
        
        # Extract the actual environment from the dict structure
        # For libero: dict[suite_name][task_id] -> vec_env
        suite_name = list(env_dict.keys())[0]
        task_id = list(env_dict[suite_name].keys())[0]
        self.vec_env = env_dict[suite_name][task_id]
        self.env = self.vec_env.envs[0]  # Get the first environment from vector env
        
        # Get max episode steps from the environment (for episode termination tracking)
        if hasattr(self.env, '_max_episode_steps'):
            self.max_episode_steps = self.env._max_episode_steps
            self.logger.info(f"Max episode steps: {self.max_episode_steps}")
        else:
            self.max_episode_steps = config.env.episode_length
            self.logger.warning(f"Environment doesn't have _max_episode_steps, using episode_length: {self.max_episode_steps}")
        
        # Build lerobot features from environment config
        lerobot_features = _env_obs_to_lerobot_features(config.env)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
            rename_map=config.env.features_map,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Simulation environment ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing
        
        # Flag to signal policy reset on next observation
        self._reset_policy_on_next_obs = True  # Reset at the start
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_steps = []
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        
        # Video recording
        if config.save_videos:
            self.videos_dir = Path(config.videos_dir)
            self.videos_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Videos will be saved to: {self.videos_dir}")
        else:
            self.videos_dir = None
        self.n_episodes_rendered = 0
        self.current_episode_frames = []

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the simulation client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the simulation client"""
        self.shutdown_event.set()

        self.vec_env.close()
        self.logger.debug("Environment closed")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run SimClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep}")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Processes incoming actions and adds them to the queue.
        
        Behavior:
        - If replace_actions_on_new=True: Replace ALL remaining actions with new ones
        - If replace_actions_on_new=False: Only add new timesteps, aggregate overlapping ones
        """
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        
        with self.latest_action_lock:
            latest_action = self.latest_action
        
        # If replace_actions_on_new is True, discard all old actions and use only new ones
        if self.config.replace_actions_on_new:
            with self.action_queue_lock:
                old_queue_size = self.action_queue.qsize()
            
            # Add all new actions that are newer than the latest executed action
            for new_action in incoming_actions:
                if new_action.get_timestep() > latest_action:
                    future_action_queue.put(new_action)
            
            if old_queue_size > 0:
                self.logger.debug(
                    f"Replaced {old_queue_size} old actions with {future_action_queue.qsize()} new actions "
                    f"(replace_actions_on_new=True)"
                )
        else:
            # Original behavior: aggregate overlapping actions, keep old ones
            with self.action_queue_lock:
                internal_queue = self.action_queue.queue
            
            current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

            for new_action in incoming_actions:
                # New action is older than the latest action in the queue, skip it
                if new_action.get_timestep() <= latest_action:
                    continue

                # If the new action's timestep is not in the current action queue, add it directly
                elif new_action.get_timestep() not in current_action_queue:
                    future_action_queue.put(new_action)
                    continue

                # If the new action's timestep is in the current action queue, aggregate it
                future_action_queue.put(
                    TimedAction(
                        timestamp=new_action.get_timestamp(),
                        timestep=new_action.get_timestep(),
                        action=aggregate_fn(
                            current_action_queue[new_action.get_timestep()], new_action.get_action()
                        ),
                    )
                )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use GetActions to get actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start
                
                # Only use the first N actions from the chunk (discard the rest for more reactive behavior)
                original_count = len(timed_actions)
                if len(timed_actions) > self.config.max_actions_to_use:
                    timed_actions = timed_actions[:self.config.max_actions_to_use]
                    self.logger.debug(
                        f"Truncated action chunk from {original_count} to {len(timed_actions)} actions "
                        f"(max_actions_to_use={self.config.max_actions_to_use})"
                    )

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                # Check if this is a normal shutdown (channel closed)
                if not self.running or e.code() == grpc.StatusCode.CANCELLED:
                    self.logger.debug("Action receiver thread shutting down (channel closed)")
                    break
                else:
                    self.logger.error(f"Error receiving actions: {e}")
        
        self.logger.info("Action receiving thread stopped")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_array(self, action_tensor: torch.Tensor) -> np.ndarray:
        """Convert action tensor to numpy array for environment step."""
        return action_tensor.cpu().numpy()

    def _render_frame(self):
        """Render and capture current frame from environment."""
        if self.videos_dir is not None and self.n_episodes_rendered < self.config.max_episodes_rendered:
            try:
                # Render frame from the environment
                frames = self.vec_env.call("render")
                if frames and len(frames) > 0:
                    # frames is a list of frames for each env in VectorEnv
                    # We only have 1 env, so take the first
                    frame = frames[0]
                    self.current_episode_frames.append(frame)
            except Exception as e:
                self.logger.warning(f"Failed to render frame: {e}")
    
    def _save_episode_video(self, episode_num: int, steps: int, success: bool):
        """Save video for the completed episode."""
        if not self.current_episode_frames:
            self.logger.warning(f"No frames to save for episode {episode_num}")
            return
        
        # Create video filename
        status = "success" if success else "fail"
        video_filename = f"episode_{episode_num:03d}_{status}_steps_{steps}.mp4"
        video_path = self.videos_dir / video_filename
        
        try:
            # Stack frames: list of (H, W, C) -> (T, H, W, C)
            frames_array = np.stack(self.current_episode_frames, axis=0)
            
            # Save video in a separate thread to avoid blocking
            thread = threading.Thread(
                target=write_video,
                args=(str(video_path), frames_array, self.config.fps),
                daemon=True,
            )
            thread.start()
            
            self.logger.info(f"Saving video: {video_path} ({len(self.current_episode_frames)} frames)")
        except Exception as e:
            self.logger.error(f"Failed to save video for episode {episode_num}: {e}")

    def control_loop_action(self, verbose: bool = False) -> tuple[np.ndarray, float, bool, dict]:
        """Execute action from queue in the environment.
        
        Returns:
            observation, reward, done, info from environment step
        """
        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        # Convert action tensor to numpy array
        action_array = self._action_tensor_to_action_array(timed_action.get_action())
        
        # Execute action in environment
        obs, reward, terminated, truncated, info = self.vec_env.step(np.array([action_array]))
        
        # Render frame after action (for video recording)
        self._render_frame()
        
        # Track episode steps BEFORE incrementing (since we increment after this function)
        will_exceed_max_steps = (self.current_episode_steps + 1) >= self.max_episode_steps
        
        # Check if episode is done based on multiple signals
        done = terminated[0] or truncated[0]
        
        # Enforce max episode steps limit ourselves (Libero doesn't always do this correctly)
        if will_exceed_max_steps and not done:
            done = True
            self.logger.info(
                f"Episode reached max steps ({self.max_episode_steps}), forcing termination"
            )
            # Mark as truncated in info
            if "is_success" not in info or not info["is_success"]:
                info["truncated_by_max_steps"] = True
        
        # Extract from vectorized format (handles nested dicts)
        obs = self._extract_single_obs(obs)
        reward = reward[0]
        # info might be a dict or a list depending on the environment
        if isinstance(info, (list, tuple)):
            info = info[0]
        # If info is already a dict, keep it as is
        
        # Additional safety check: if info indicates the episode is truly done (from final_info)
        # This catches cases where Libero's internal state says done but terminated/truncated are False
        if "final_info" in info and not done:
            done = True
            self.logger.debug(f"Detected episode end via final_info: {info['final_info']}")
        
        # Track episode reward
        self.current_episode_reward += reward

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Reward: {reward:.4f} | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return obs, reward, done, info

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation.
        
        Uses request_new_at parameter: when queue size drops to this value, request new actions.
        """
        with self.action_queue_lock:
            current_queue_size = self.action_queue.qsize()
            # Request new observation when queue size drops to request_new_at
            return current_queue_size <= self.config.request_new_at

    def control_loop_observation(self, obs: dict, task: str, verbose: bool = False) -> RawObservation:
        """Process and send observation to policy server."""
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            # Format environment observation to robot observation format
            raw_observation: RawObservation = _format_env_observation(obs, self.config.env, task)

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
                reset_policy=self._reset_policy_on_next_obs,  # Signal to reset policy state
            )
            
            # Clear the flag after setting it in observation
            if self._reset_policy_on_next_obs:
                self._reset_policy_on_next_obs = False

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                # Only clear must_go if queue is NOT empty (i.e., we have received actions)
                # If queue is still empty, keep must_go set so server will process the observation
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            # Only clear must_go after we've sent an observation AND we have actions in the queue
            # This prevents clearing the flag before the first actions arrive
            if observation.must_go and current_queue_size > 0:
                # must-go event will be set again after queue becomes empty
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Processing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def _extract_single_obs(self, obs: dict) -> dict:
        """Extract single observation from vectorized format.
        
        Handles nested dict structure (e.g., Libero's pixels dict).
        """
        result = {}
        for k, v in obs.items():
            if isinstance(v, dict):
                # Recursively handle nested dicts
                result[k] = self._extract_single_obs(v)
            else:
                # Extract first element from batch
                result[k] = v[0]
        return result

    def reset_environment(self, seed: int | None = None):
        """Reset the simulation environment.
        
        Args:
            seed: Random seed for reproducibility. If None, uses random initialization.
        """
        obs, info = self.vec_env.reset(seed=[seed] if seed is not None else None)
        # Extract from vectorized format (handles nested dicts)
        obs = self._extract_single_obs(obs)
        # info might be a dict or a list depending on the environment
        if isinstance(info, (list, tuple)) and len(info) > 0:
            info = info[0]
        # If info is already a dict, keep it as is
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        
        # Mark that we need to reset policy state for the next observation
        self._reset_policy_on_next_obs = True
        
        # Reset frame buffer for new episode
        self.current_episode_frames = []
        
        # Render initial frame
        self._render_frame()
        
        self.logger.debug(
            f"Environment reset (seed={seed}). Max steps for this episode: {self.max_episode_steps}"
        )
        return obs, info

    def control_loop(self, task: str, verbose: bool = False):
        """Combined function for executing actions and streaming observations in simulation."""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        episode_count = 0

        while self.running and episode_count < self.config.n_episodes:
            # Calculate seed for this episode (if seed is provided)
            episode_seed = self.config.seed + episode_count if self.config.seed is not None else None
            
            # Reset environment at the start of each episode (with deterministic seed)
            obs, info = self.reset_environment(seed=episode_seed)
            episode_done = False
            step_count = 0
            
            self.logger.info(
                f"Starting episode {episode_count + 1}/{self.config.n_episodes} (seed={episode_seed})"
            )
            
            # Create progress bar for this episode
            pbar = tqdm(
                total=self.max_episode_steps,
                desc=f"Episode {episode_count + 1}/{self.config.n_episodes}",
                unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                leave=True,
            )
            
            # Episode loop: run until episode is done
            while not episode_done and self.running:
                control_loop_start = time.perf_counter()
                
                # (1) Send observation if ready
                if self._ready_to_send_observation():
                    self.control_loop_observation(obs, task, verbose)
                
                # (2) Perform action if available
                if self.actions_available():
                    obs, reward, done, info = self.control_loop_action(verbose)
                    step_count += 1
                    self.current_episode_steps += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'reward': f'{self.current_episode_reward:.2f}',
                        'queue': self.action_queue.qsize()
                    })
                    
                    # Check if episode is done
                    if done:
                        episode_done = True
                        episode_count += 1
                        self.episode_rewards.append(self.current_episode_reward)
                        self.episode_steps.append(self.current_episode_steps)
                        
                        # Check if the episode was successful
                        # Convert numpy types to Python native types for proper formatting
                        is_success = bool(info.get("is_success", False))
                        self.episode_successes.append(is_success)
                        
                        # Convert episode metrics to Python native types
                        episode_reward = float(self.current_episode_reward)
                        episode_steps = int(self.current_episode_steps)
                        
                        # Calculate running statistics
                        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                        success_rate = sum(self.episode_successes) / len(self.episode_successes)
                        avg_steps = sum(self.episode_steps) / len(self.episode_steps)
                        successful_count = int(sum(self.episode_successes))
                        
                        # Update progress bar with final status
                        pbar.set_postfix({
                            'reward': f'{episode_reward:.2f}',
                            'success': '✓' if is_success else '✗',
                            'rate': f'{success_rate:.1%}'
                        })
                        pbar.close()
                        
                        self.logger.info(
                            f"\n{'='*70}\n"
                            f"Episode {episode_count}/{self.config.n_episodes} finished | "
                            f"Steps: {episode_steps} | "
                            f"Reward: {episode_reward:.4f} | "
                            f"Success: {'✓' if is_success else '✗'}\n"
                            f"{'-'*70}\n"
                            f"Running Stats (avg over {episode_count} episodes):\n"
                            f"  Success Rate: {success_rate:.1%} ({successful_count}/{episode_count})\n"
                            f"  Avg Reward:   {avg_reward:.4f}\n"
                            f"  Avg Steps:    {avg_steps:.1f}\n"
                            f"{'='*70}"
                        )
                        
                        # Clear the action queue when episode ends
                        # (remaining actions are for the old episode and shouldn't be executed in the new one)
                        with self.action_queue_lock:
                            old_queue_size = self.action_queue.qsize()
                            self.action_queue = Queue()
                            if old_queue_size > 0:
                                self.logger.debug(f"Cleared {old_queue_size} remaining actions from queue")
                        
                        # Reset latest_action counter for the new episode
                        with self.latest_action_lock:
                            self.latest_action = -1
                        
                        # Save video for this episode
                        if self.videos_dir is not None and self.n_episodes_rendered < self.config.max_episodes_rendered:
                            self._save_episode_video(episode_count, episode_steps, is_success)
                            self.n_episodes_rendered += 1

                self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
                # Dynamically adjust sleep time to maintain the desired control frequency
                time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))
            
            # Make sure progress bar is closed even if episode was interrupted
            if not episode_done:
                pbar.close()
        
        # All episodes completed, signal shutdown
        self.logger.info(f"All {self.config.n_episodes} episodes completed, shutting down...")
        self.shutdown_event.set()

        # Log final statistics
        if self.episode_rewards:
            # Convert all numpy types to Python native types
            avg_reward = float(sum(self.episode_rewards) / len(self.episode_rewards))
            success_rate = float(sum(self.episode_successes) / len(self.episode_successes)) if self.episode_successes else 0.0
            avg_steps = float(sum(self.episode_steps) / len(self.episode_steps)) if self.episode_steps else 0.0
            min_reward = float(min(self.episode_rewards))
            max_reward = float(max(self.episode_rewards))
            min_steps = int(min(self.episode_steps)) if self.episode_steps else 0
            max_steps = int(max(self.episode_steps)) if self.episode_steps else 0
            successful_count = int(sum(self.episode_successes))
            total_count = int(episode_count)
            
            self.logger.info(
                f"\n{'='*70}\n"
                f"🎉 Evaluation Complete! 🎉\n"
                f"{'='*70}\n"
                f"Total Episodes: {total_count}\n"
                f"\n"
                f"Success Metrics:\n"
                f"  Success Rate:     {success_rate:.1%} ({successful_count}/{total_count})\n"
                f"  Successful Eps:   {successful_count}\n"
                f"  Failed Eps:       {total_count - successful_count}\n"
                f"\n"
                f"Reward Statistics:\n"
                f"  Average Reward:   {avg_reward:.4f}\n"
                f"  Min Reward:       {min_reward:.4f}\n"
                f"  Max Reward:       {max_reward:.4f}\n"
                f"\n"
                f"Step Statistics:\n"
                f"  Average Steps:    {avg_steps:.1f}\n"
                f"  Min Steps:        {min_steps}\n"
                f"  Max Steps:        {max_steps}\n"
                f"\n"
                f"Episode Details:\n"
            )
            
            # Print individual episode results
            for i, (reward, success, steps) in enumerate(zip(self.episode_rewards, self.episode_successes, self.episode_steps), 1):
                # Convert to Python native types for formatting
                reward = float(reward)
                success = bool(success)
                steps = int(steps)
                status = "✓" if success else "✗"
                self.logger.info(
                    f"  Ep {i:2d}: {status} | Reward: {reward:6.3f} | Steps: {steps:3d}"
                )
            
            self.logger.info(f"{'='*70}\n")


@draccus.wrap()
def async_sim_client(cfg: SimClientConfig):
    """Main entry point for simulation client."""
    logging.info(pformat(asdict(cfg)))

    if cfg.env.type not in SUPPORTED_ENVS:
        raise ValueError(f"Environment {cfg.env.type} not yet supported! Supported: {SUPPORTED_ENVS}")

    client = SimClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(
            target=client.receive_actions,
            daemon=True
        )

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_sim_client()  # run the client

