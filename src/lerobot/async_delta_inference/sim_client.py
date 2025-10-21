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
    from lerobot.configs.types import FeatureType
    from lerobot.utils.constants import OBS_STR

    # Build the features dictionary in the LeRobot dataset format (with dtype, shape, names)
    features = {}

    for key, policy_ft in env_config.features.items():
        if key == "action":
            continue

        # Convert PolicyFeature to dataset feature format
        if policy_ft.type == FeatureType.VISUAL:
            # For images: dtype is "image" or "video", shape is (H, W, C)
            lerobot_key = env_config.features_map.get(key, f"{OBS_STR}.images.{key}")
            features[lerobot_key] = {
                "dtype": "image",
                "shape": policy_ft.shape,  # Already in (H, W, C) format from env config
                "names": ["height", "width", "channels"],
            }
        elif policy_ft.type == FeatureType.STATE:
            # For state: dtype is "float32", shape is (state_dim,)
            lerobot_key = env_config.features_map.get(key, f"{OBS_STR}.state")
            features[lerobot_key] = {
                "dtype": "float32",
                "shape": policy_ft.shape,
                "names": [key],  # Single name pointing to the state vector in raw obs
            }

    return features


def _format_env_observation(obs: dict, env_config, task: str = "") -> RawObservation:
    """Format environment observation to match robot observation format.
    
    Converts Gym environment observation format to raw format with simple key names.
    The conversion to LeRobot format happens on the server side.
    
    For Libero, this means:
    - 'pixels/agentview_image' -> 'agentview_image' (simple camera name)
    - 'pixels/robot0_eye_in_hand_image' -> 'robot0_eye_in_hand_image'
    - 'agent_pos' -> 'agent_pos' (keep as vector)
    """
    raw_obs: RawObservation = {}
    
    # Process images - use simple camera names
    if "pixels" in obs:
        for cam_name, img in obs["pixels"].items():
            # Use simple camera name without LeRobot prefix
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            raw_obs[cam_name] = img
    
    # Process state - keep as vector
    if "agent_pos" in obs:
        state = obs["agent_pos"]
        if isinstance(state, np.ndarray):
            state = state  # Keep as numpy array
        elif isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        raw_obs["agent_pos"] = state
    
    # Add task if provided
    if task:
        raw_obs["task"] = task
    
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
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_successes = []
        self.current_episode_reward = 0.0

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
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

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
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

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
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_array(self, action_tensor: torch.Tensor) -> np.ndarray:
        """Convert action tensor to numpy array for environment step."""
        return action_tensor.cpu().numpy()

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
        done = terminated[0] or truncated[0]
        
        # Extract from vectorized format - handle nested pixels dict
        unwrapped_obs = {}
        for k, v in obs.items():
            if k == "pixels" and isinstance(v, dict):
                # pixels is a nested dict, unwrap each camera image
                unwrapped_obs[k] = {cam_name: img[0] for cam_name, img in v.items()}
            else:
                unwrapped_obs[k] = v[0]
        obs = unwrapped_obs
        reward = reward[0]
        info = info[0] if isinstance(info, (list, tuple)) else info
        
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
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, obs: dict, task: str, verbose: bool = False) -> RawObservation:
        """Process and send observation to policy server."""
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            # Format environment observation to robot observation format
            raw_observation: RawObservation = _format_env_observation(obs, self.config.env, self.env.task_description)

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
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

    def reset_environment(self):
        """Reset the simulation environment."""
        obs, info = self.vec_env.reset()
        # Extract from vectorized format - handle nested pixels dict
        unwrapped_obs = {}
        for k, v in obs.items():
            if k == "pixels" and isinstance(v, dict):
                # pixels is a nested dict, unwrap each camera image
                unwrapped_obs[k] = {cam_name: img[0] for cam_name, img in v.items()}
            else:
                unwrapped_obs[k] = v[0]
        obs = unwrapped_obs
        info = info[0] if isinstance(info, (list, tuple)) else info
        self.current_episode_reward = 0.0
        return obs, info

    def control_loop(self, task: str, verbose: bool = False):
        """Combined function for executing actions and streaming observations in simulation."""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        # Reset environment to start
        obs, info = self.reset_environment()
        
        episode_count = 0
        step_count = 0
        
        # Get max steps from environment config
        max_steps = getattr(self.config.env, 'episode_length', None)
        pbar = None

        while self.running and episode_count < self.config.n_episodes:
            control_loop_start = time.perf_counter()
            
            # (1) Send observation if ready
            if self._ready_to_send_observation():
                self.control_loop_observation(obs, task, verbose)
            
            # (2) Perform action if available
            if self.actions_available():
                obs, reward, done, info = self.control_loop_action(verbose)
                step_count += 1
                
                # Create progress bar for first step of episode
                if pbar is None:
                    pbar = tqdm(
                        total=max_steps,
                        desc=f"Episode {episode_count + 1}/{self.config.n_episodes}",
                        unit="step"
                    )
                
                # Update progress bar
                pbar.update(1)
                
                # Check if episode should end: either environment says done, or reached max steps
                max_steps_reached = (
                    hasattr(self.config.env, 'episode_length') 
                    and step_count >= self.config.env.episode_length
                )
                episode_done = done or max_steps_reached
                
                # Check if episode is done
                if episode_done:
                    episode_count += 1
                    self.episode_rewards.append(self.current_episode_reward)
                    
                    # Check if the episode was successful
                    is_success = info.get("is_success", False)
                    self.episode_successes.append(is_success)
                    
                    # Add reason for episode end
                    end_reason = "max_steps" if max_steps_reached and not done else ("success" if is_success else "done")
                    
                    # Close progress bar
                    if pbar is not None:
                        pbar.close()
                        pbar = None
                    
                    self.logger.info(
                        f"Episode {episode_count}/{self.config.n_episodes} finished ({end_reason}) | "
                        f"Steps: {step_count} | "
                        f"Reward: {self.current_episode_reward:.4f} | "
                        f"Success: {is_success}"
                    )
                    
                    step_count = 0
                    
                    # Reset environment for next episode if not done with all episodes
                    if episode_count < self.config.n_episodes:
                        obs, info = self.reset_environment()

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))
        
        # Close progress bar if still open
        if pbar is not None:
            pbar.close()

        # Log final statistics
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            success_rate = sum(self.episode_successes) / len(self.episode_successes) if self.episode_successes else 0.0
            
            self.logger.info(
                f"\n{'='*60}\n"
                f"Evaluation complete!\n"
                f"Episodes: {episode_count}\n"
                f"Average Reward: {avg_reward:.4f}\n"
                f"Success Rate: {success_rate:.2%}\n"
                f"{'='*60}"
            )


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
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

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

