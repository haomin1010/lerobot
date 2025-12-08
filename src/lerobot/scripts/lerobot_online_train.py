#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Online training script that collects data from environment and trains the policy.

This script combines data collection (similar to lerobot_eval.py) with training (similar to lerobot_train.py).
It alternates between:
1. Collecting episodes from the environment using the current policy
2. Adding collected data to the dataset (placeholder for user implementation)
3. Training the policy on the accumulated dataset

Usage examples:

```
lerobot-online-train \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --dataset.repo_id=my_online_dataset \
    --online.collect_episodes_per_iteration=10 \
    --online.train_steps_per_iteration=1000 \
    --online.n_iterations=100 \
    --policy.use_amp=false \
    --policy.device=cuda
```
"""

import logging
import time
from contextlib import nullcontext
from collections.abc import Callable
from pathlib import Path
from pprint import pformat
from typing import Any
from tqdm import trange
from copy import deepcopy

import einops
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.configs import parser
from lerobot.configs.train import OnlineTrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import _compile_episode_data
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)


def rollout(
        env: gym.vector.VectorEnv,
        policy: PreTrainedPolicy,
        env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
        seeds: list[int] | None = None,
        return_observations: bool = False,
        render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        # This handles nested dictionaries (e.g., robot_state -> state)
        observation = env_preprocessor(observation)

        # Save observation AFTER env_preprocessor (nested dicts are flattened)
        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        action_transition = {"action": action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition["action"]

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        observation = add_envs_task(env, observation)
        observation = env_preprocessor(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        # Only stack keys that start with "observation."
        for key in all_observations[0]:
            if key.startswith(f"{OBS_STR}."):
                stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    cmp=False,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        cmp: Whether to use CMP (Contrastive Model Prediction) loss. Defaults to False.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch, cmp=cmp)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def collect_episodes(
    env,
    policy: PreTrainedPolicy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    start_seed: int | None = None,
    start_episode_index: int = 0,
) -> dict:
    """Collect episodes from the environment using the current policy.

    Args:
        env: The batch of environments.
        policy: The policy to use for action selection.
        env_preprocessor: Environment-specific preprocessor.
        env_postprocessor: Environment-specific postprocessor.
        preprocessor: Policy preprocessor.
        postprocessor: Policy postprocessor.
        n_episodes: Number of episodes to collect.
        start_seed: Starting seed for environment reset.
        start_episode_index: Starting episode index for the collected episodes.

    Returns:
        Dictionary containing compiled episode data ready to be added to dataset.
    """
    # Determine how many batched rollouts we need
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    all_episode_data = []
    current_episode_index = start_episode_index
    current_data_index = 0

    for batch_ix in range(n_batches):
        # Calculate how many episodes to collect in this batch
        episodes_this_batch = min(env.num_envs, n_episodes - batch_ix * env.num_envs)

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs),
                start_seed + (batch_ix * env.num_envs) + episodes_this_batch,
            )

        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=True,  # We need observations for dataset
            render_callback=None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered
        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Only process the episodes we actually need
        if episodes_this_batch < env.num_envs:
            # Slice to only include the episodes we need
            rollout_data = {
                k: v[:episodes_this_batch] if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                for k, v in rollout_data.items()
            }
            done_indices = done_indices[:episodes_this_batch]

        # Compile episode data for this batch
        batch_episode_data = _compile_episode_data(
            rollout_data,
            done_indices,
            start_episode_index=current_episode_index,
            start_data_index=current_data_index,
            fps=env.unwrapped.metadata["render_fps"],
        )

        all_episode_data.append(batch_episode_data)
        current_episode_index += episodes_this_batch
        # Update data index for next batch
        if len(batch_episode_data.get("index", [])) > 0:
            current_data_index = batch_episode_data["index"][-1].item() + 1

    # Concatenate all episode data
    if len(all_episode_data) > 1:
        combined_data = {}
        for key in all_episode_data[0]:
            combined_data[key] = torch.cat([ep[key] for ep in all_episode_data])
        return combined_data
    elif len(all_episode_data) == 1:
        return all_episode_data[0]
    else:
        return {}


def add_episodes_to_dataset(
    online_dataset: LeRobotDataset,
    episode_data: dict,
) -> None:
    """Add collected episodes to the dataset.

    This function converts episode_data from batch format to individual frames
    and adds them to the dataset using add_frame() and save_episode().

    Args:
        online_dataset: The online dataset to add episodes to.
        episode_data: Dictionary containing episode data from collect_episodes.
            Expected keys:
            - ACTION: actions (total_frames, action_dim)
            - REWARD: rewards (total_frames,)
            - DONE: done flags (total_frames,)
            - episode_index: episode indices (total_frames,)
            - frame_index: frame indices within episodes (total_frames,)
            - timestamp: timestamps (total_frames,)
            - index: global frame indices (total_frames,)
            - observation.*: observation keys (total_frames, ...)
            - next.success: success flags (total_frames,) [optional]
            - task: task names (total_frames,) [optional, if not in observation]
    """
    if not episode_data:
        logging.warning("episode_data is empty, nothing to add to dataset")
        return

    # Get total number of frames
    total_frames = len(episode_data[ACTION])

    # Track current episode to detect episode boundaries
    current_episode_index = None

    # Extract task information if available
    # Task might be in episode_data directly, or in observation keys
    task_key = None
    if "task" in episode_data:
        task_key = "task"
    else:
        # Check if task is in observation keys (it might be saved as "observation.task" or just "task")
        for key in episode_data:
            if key == "task" or key.endswith(".task"):
                task_key = key
                break

    # Iterate through all frames
    for frame_idx in range(total_frames):
        # Create frame dictionary
        frame_dict = {}

        # Add action (reward and done are not in dataset features, so we don't add them)
        frame_dict[ACTION] = episode_data[ACTION][frame_idx]

        # Add task (required by add_frame)
        if task_key and task_key in episode_data:
            task_value = episode_data[task_key][frame_idx]
            # Task might be a list (from add_envs_task) or a string
            if isinstance(task_value, (list, tuple)) and len(task_value) > 0:
                frame_dict["task"] = task_value[0] if isinstance(task_value[0], str) else str(task_value[0])
            elif isinstance(task_value, torch.Tensor):
                # Convert tensor to string if needed
                task_item = task_value.item() if task_value.numel() == 1 else str(task_value)
                frame_dict["task"] = str(task_item) if not isinstance(task_item, str) else task_item
            else:
                frame_dict["task"] = str(task_value) if task_value else ""
        else:
            # Default task if not available
            frame_dict["task"] = ""

        # Don't add timestamp - it's in DEFAULT_FEATURES and will be handled by add_frame automatically

        # Add all observation keys (only those starting with "observation.")
        for key in episode_data:
            if key.startswith(f"{OBS_STR}."):
                value = episode_data[key][frame_idx]
                
                # Convert images from channel-first (C, H, W) to channel-last (H, W, C) if needed
                # Dataset features expect channel-last format (H, W, C) based on names=["height", "width", "channel"]
                if key in online_dataset.features:
                    feat = online_dataset.features[key]
                    if feat.get("dtype") in ["image", "video"]:
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().numpy()
                        if isinstance(value, np.ndarray) and value.ndim == 3:
                            # Check if it's channel-first (C, H, W) - typically C=3 is small
                            if value.shape[0] == 3 and value.shape[0] < value.shape[1] and value.shape[0] < value.shape[2]:
                                # Convert from (C, H, W) to (H, W, C)
                                value = np.transpose(value, (1, 2, 0))
                        
                        # Log image shape for debugging (first frame only)
                        if frame_idx == 0:
                            logging.info(
                                f"Image feature '{key}': actual_shape={value.shape if isinstance(value, np.ndarray) else type(value)}, "
                                f"expected_shape={feat.get('shape')}, names={feat.get('names')}, dtype={feat.get('dtype')}"
                            )
                
                frame_dict[key] = value

        # Don't add next.success as complementary_info if it's not in dataset features
        # Only add complementary_info keys if they exist in dataset features
        # if "next.success" in episode_data:
        #     complementary_info_keys = [k for k in online_dataset.features if k.startswith("complementary_info.")]
        #     if "complementary_info.success" in online_dataset.features:
        #         success_value = episode_data["next.success"][frame_idx]
        #         if isinstance(success_value, torch.Tensor):
        #             if success_value.numel() == 1:
        #                 frame_dict["complementary_info.success"] = success_value.item()
        #             else:
        #                 frame_dict["complementary_info.success"] = success_value
        #         else:
        #             frame_dict["complementary_info.success"] = success_value

        # Log frame_dict keys and dataset features for first frame
        if frame_idx == 0:
            logging.info(f"Frame dict keys: {sorted(frame_dict.keys())}")
            logging.info(f"Dataset expected features (excluding DEFAULT_FEATURES): {sorted(set(online_dataset.features.keys()) - {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'})}")
            logging.info(f"Extra keys in frame_dict (not in dataset features): {sorted(set(frame_dict.keys()) - {'task'} - set(online_dataset.features.keys()))}")

        # Check if we need to save episode before adding frame (episode boundary)
        episode_index = episode_data["episode_index"][frame_idx].item()
        # Get done flag from episode_data (not from frame_dict, as it's not in dataset features)
        done = episode_data[DONE][frame_idx] if DONE in episode_data else False

        # Save previous episode if episode index changed (new episode started)
        if current_episode_index is not None and episode_index != current_episode_index:
            if online_dataset.episode_buffer is not None and online_dataset.episode_buffer["size"] > 0:
                online_dataset.save_episode()

        # Add frame to dataset
        online_dataset.add_frame(frame_dict)

        # Save episode if done flag is True (episode ended)
        if isinstance(done, torch.Tensor):
            if done.item():
                online_dataset.save_episode()
        elif done:
            online_dataset.save_episode()

        # Update current episode index
        current_episode_index = episode_index

    # Save the last episode if there are any remaining frames
    if total_frames > 0:
        if online_dataset.episode_buffer is not None and online_dataset.episode_buffer["size"] > 0:
            online_dataset.save_episode()

    logging.info(f"Added {total_frames} frames to dataset")


@parser.wrap()
def online_train_main(cfg: OnlineTrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function for online training with data collection.

    This function orchestrates the online training pipeline:
    1. Sets up environment, policy, and dataset
    2. Alternates between collecting episodes and training
    3. Periodically saves checkpoints and logs metrics

    Args:
        cfg: An `OnlineTrainPipelineConfig` object containing all configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create environment for data collection
    if is_main_process:
        logging.info("Creating environment for data collection")
    collect_env_dict = make_env(
        cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs
    )
    # Keep the full dict to use all tasks, not just the first one
    # Structure: {suite_name: {task_id: vec_env}}

    # Create or load datasets
    offline_dataset = None
    online_dataset = None
    
    if is_main_process:
        logging.info("Creating/loading datasets")
        
        # Load offline dataset if specified
        if cfg.online.start_with_offline_dataset:
            logging.info("Loading offline dataset for training")
            offline_dataset = make_dataset(cfg)
            logging.info(
                f"Offline dataset loaded: {offline_dataset.num_episodes} episodes, "
                f"{offline_dataset.num_frames} frames"
            )
        
        # Create or load online dataset for collecting new episodes
        online_dataset_repo_id = cfg.online.online_dataset_repo_id or cfg.dataset.repo_id
        online_dataset_root = cfg.online.online_dataset_root or cfg.dataset.root
        
        # Check if online dataset already exists by checking for info.json
        online_dataset_path = Path(online_dataset_root) / online_dataset_repo_id
        online_meta_path = online_dataset_path / "meta" / "info.json"
        
        if online_meta_path.exists():
            # Dataset exists, load it
            logging.info(f"Loading existing online dataset: {online_dataset_repo_id}")
            online_dataset = LeRobotDataset(
                online_dataset_repo_id,
                root=online_dataset_root,
                video_backend=cfg.dataset.video_backend,
            )
            logging.info(
                f"Online dataset loaded: {online_dataset.num_episodes} episodes, "
                f"{online_dataset.num_frames} frames"
            )
        else:
            # Dataset doesn't exist, create new one
            logging.info(f"Creating new online dataset: {online_dataset_repo_id}")
            
            # Get configuration from offline dataset if available, otherwise from environment
            if offline_dataset is not None:
                # Copy configuration from offline dataset
                fps = offline_dataset.meta.fps
                features = offline_dataset.meta.features.copy()
                robot_type = offline_dataset.meta.robot_type
                logging.info(
                    f"Using configuration from offline dataset: fps={fps}, "
                )
            else:
                # Get configuration from environment
                fps = cfg.env.fps
                # Convert environment features to dataset features format
                from lerobot.envs.utils import env_to_policy_features
                from lerobot.datasets.utils import dataset_to_policy_features
                from lerobot.utils.constants import ACTION, REWARD, DONE
                
                # Get policy features from env config
                policy_features = env_to_policy_features(cfg.env)
                
                # Convert to dataset features format
                features = {}
                for key, feat in policy_features.items():
                    if feat.type.name == "VISUAL":
                        # For visual features, use video dtype and keep channel-last shape
                        # Note: env features are already in channel-last format (h, w, c)
                        features[key] = {
                            "dtype": "video",
                            "shape": feat.shape,  # (h, w, c)
                            "names": ["height", "width", "channels"],
                        }
                    elif feat.type.name == "STATE" or feat.type.name == "ACTION":
                        features[key] = {
                            "dtype": "float32",
                            "shape": feat.shape,
                            "names": None,
                        }
                    else:
                        features[key] = {
                            "dtype": "float32",
                            "shape": feat.shape,
                            "names": None,
                        }
                
                # Add default features (reward, done)
                features[REWARD] = {"dtype": "float32", "shape": (1,), "names": None}
                features[DONE] = {"dtype": "bool", "shape": (1,), "names": None}
                
                robot_type = None
                logging.info(
                    f"Using configuration from environment: fps={fps}, "
                )
            
            # Create empty online dataset
            online_dataset = LeRobotDataset.create(
                online_dataset_repo_id,
                fps=fps,
                features=features,
                root=online_dataset_root,
                robot_type=robot_type,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size if hasattr(cfg.dataset, "video_encoding_batch_size") else 1,
            )
            logging.info("Empty online dataset created successfully")

    accelerator.wait_for_everyone()

    if not is_main_process:
        # Load datasets on non-main processes
        if cfg.online.start_with_offline_dataset:
            offline_dataset = make_dataset(cfg)
        
        online_dataset_repo_id = cfg.online.online_dataset_repo_id or cfg.dataset.repo_id
        online_dataset_root = cfg.online.online_dataset_root or cfg.dataset.root
        try:
            online_dataset = LeRobotDataset(
                online_dataset_repo_id,
                root=online_dataset_root,
                video_backend=cfg.dataset.video_backend,
            )
        except (FileNotFoundError, NotADirectoryError):
            # This shouldn't happen if main process created it, but handle gracefully
            raise RuntimeError(
                f"Online dataset {online_dataset_repo_id} not found. "
                "It should have been created by the main process."
            )
    
    # Use offline dataset for training if available, otherwise use online dataset
    # The online dataset will be used for collecting new episodes
    training_dataset = offline_dataset if offline_dataset is not None else online_dataset

    # Create policy
    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=training_dataset.meta,
        rename_map=cfg.rename_map,
    )

    accelerator.wait_for_everyone()

    # Create processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = training_dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": training_dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": training_dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # Create environment-specific processors
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env)

    # Create separate preprocessor for data collection (without normalizer)
    # This matches lerobot_eval.py's approach - we want raw data, not normalized data
    collect_preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    collect_preprocessor, collect_postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=collect_preprocessor_overrides,
    )

    # Log processor steps for debugging
    if is_main_process:
        from lerobot.processor.normalize_processor import NormalizerProcessorStep
        processor_names = [type(step).__name__ for step in collect_preprocessor.steps]
        has_normalizer = any(isinstance(step, NormalizerProcessorStep) for step in collect_preprocessor.steps)
        logging.info(f"Collect preprocessor steps: {processor_names}")
        logging.info(f"Collect preprocessor contains normalizer_processor: {has_normalizer}")
        
        # Also log training preprocessor for comparison
        training_processor_names = [type(step).__name__ for step in preprocessor.steps]
        training_has_normalizer = any(isinstance(step, NormalizerProcessorStep) for step in preprocessor.steps)
        logging.info(f"Training preprocessor steps: {training_processor_names}")
        logging.info(f"Training preprocessor contains normalizer_processor: {training_has_normalizer}")

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # Global training step counter

    if cfg.resume:
        from lerobot.utils.train_utils import load_training_state

        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.online.n_iterations=}")
        logging.info(f"{cfg.online.collect_episodes_per_iteration=}")
        logging.info(f"{cfg.online.train_steps_per_iteration=}")
        if offline_dataset is not None:
            logging.info(
                f"Offline dataset: {offline_dataset.num_episodes} episodes, "
                f"{offline_dataset.num_frames} frames ({format_big_number(offline_dataset.num_frames)})"
            )
        logging.info(
            f"Online dataset: {online_dataset.num_episodes} episodes, "
            f"{online_dataset.num_frames} frames ({format_big_number(online_dataset.num_frames)})"
        )
        logging.info(
            f"Training dataset: {training_dataset.num_episodes} episodes, "
            f"{training_dataset.num_frames} frames ({format_big_number(training_dataset.num_frames)})"
        )
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Create dataloaders for offline and online datasets
    def create_dataloader(dataset):
        if hasattr(cfg.policy, "drop_n_last_frames"):
            shuffle = False
            sampler = EpisodeAwareSampler(
                dataset.meta.episodes["dataset_from_index"],
                dataset.meta.episodes["dataset_to_index"],
                episode_indices_to_use=dataset.episodes,
                drop_n_last_frames=cfg.policy.drop_n_last_frames,
                shuffle=True,
            )
        else:
            shuffle = True
            sampler = None

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle and not cfg.dataset.streaming,
            sampler=sampler,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

    # Create dataloader for offline dataset (if exists)
    offline_dataloader = None
    offline_dl_iter = None
    if offline_dataset is not None:
        offline_dataloader = create_dataloader(offline_dataset)
        if is_main_process:
            logging.info("Created dataloader for offline dataset")

    # Create dataloader for online dataset (only if it has data)
    online_dataloader = None
    online_dl_iter = None
    if online_dataset.num_frames > 0:
        online_dataloader = create_dataloader(online_dataset)
        if is_main_process:
            logging.info("Created dataloader for online dataset")
    else:
        if is_main_process:
            logging.info("Online dataset is empty, dataloader will be created after first data collection")

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    prepare_list = [policy, optimizer, lr_scheduler]
    if offline_dataloader is not None:
        prepare_list.append(offline_dataloader)
    if online_dataloader is not None:
        prepare_list.append(online_dataloader)
    
    prepared = accelerator.prepare(*prepare_list)
    if offline_dataloader is not None and online_dataloader is not None:
        policy, optimizer, lr_scheduler, offline_dataloader, online_dataloader = prepared
        offline_dl_iter = cycle(offline_dataloader)
        online_dl_iter = cycle(online_dataloader)
    elif offline_dataloader is not None:
        policy, optimizer, lr_scheduler, offline_dataloader = prepared
        offline_dl_iter = cycle(offline_dataloader)
    elif online_dataloader is not None:
        policy, optimizer, lr_scheduler, online_dataloader = prepared
        online_dl_iter = cycle(online_dataloader)
    else:
        policy, optimizer, lr_scheduler = prepared

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        training_dataset.num_frames,
        training_dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start online training: alternating between data collection and training")

    # Main online training loop
    for iteration in range(cfg.online.n_iterations):
        if is_main_process:
            logging.info(f"\n{'='*60}")
            logging.info(f"Iteration {iteration + 1}/{cfg.online.n_iterations}")
            logging.info(f"{'='*60}")

        # Phase 1: Collect episodes from all tasks
        if is_main_process:
            logging.info(
                f"Collecting {cfg.online.collect_episodes_per_iteration} episodes from environment"
            )
        policy.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
            # Get current episode count from online dataset to set correct episode indices
            current_episode_count = online_dataset.num_episodes if hasattr(online_dataset, "num_episodes") else 0
            
            # Collect episodes from all tasks
            all_episode_data = []
            total_tasks = sum(len(tasks) for tasks in collect_env_dict.values())
            episodes_per_task = cfg.online.collect_episodes_per_iteration // total_tasks
            remaining_episodes = cfg.online.collect_episodes_per_iteration % total_tasks
            
            current_ep_idx = current_episode_count
            task_idx = 0
            
            for suite_name, task_dict in collect_env_dict.items():
                for task_id, env in task_dict.items():
                    # Distribute remaining episodes to first few tasks
                    n_episodes_this_task = episodes_per_task + (1 if task_idx < remaining_episodes else 0)
                    
                    if n_episodes_this_task > 0:
                        if is_main_process:
                            logging.info(
                                f"Collecting {n_episodes_this_task} episodes from {suite_name} task {task_id}"
                            )
                        
                        task_episode_data = collect_episodes(
                            env=env,
                            policy=accelerator.unwrap_model(policy),
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=collect_preprocessor,  # Use collect_preprocessor (without normalizer)
                            postprocessor=collect_postprocessor,  # Use collect_postprocessor (without unnormalizer)
                            n_episodes=n_episodes_this_task,
                            start_seed=cfg.seed if cfg.seed is not None else None,
                            start_episode_index=current_ep_idx,
                        )
                        
                        if task_episode_data:
                            all_episode_data.append(task_episode_data)
                            # Update episode index for next task
                            # Count unique episode indices to get the number of episodes collected
                            if "episode_index" in task_episode_data and len(task_episode_data["episode_index"]) > 0:
                                unique_episodes = torch.unique(task_episode_data["episode_index"])
                                current_ep_idx = unique_episodes[-1].item() + 1
                            else:
                                # Fallback: increment by number of episodes collected
                                current_ep_idx += n_episodes_this_task
                    
                    task_idx += 1
            
            # Combine all episode data
            if len(all_episode_data) > 1:
                episode_data = {}
                for key in all_episode_data[0]:
                    episode_data[key] = torch.cat([ep[key] for ep in all_episode_data])
            elif len(all_episode_data) == 1:
                episode_data = all_episode_data[0]
            else:
                episode_data = {}

        # Phase 2: Add episodes to online dataset
        if is_main_process:
            logging.info("Adding collected episodes to online dataset")
            add_episodes_to_dataset(online_dataset, episode_data)

        # Close all writers to ensure parquet files are properly finalized before reading
        if is_main_process:
            online_dataset._close_writer()
            if hasattr(online_dataset.meta, "_close_writer"):
                online_dataset.meta._close_writer()

        # Wait for dataset update to complete
        accelerator.wait_for_everyone()

        # Reload online dataset to include new episodes
        # Note: The online_dataset should be updated in-place by add_episodes_to_dataset
        # If the dataset implementation requires reloading, do it here
        # Force reload dataset metadata if needed
        if hasattr(online_dataset.meta, "reload"):
            online_dataset.meta.reload()
        
        # Recreate dataloaders with updated datasets
        if offline_dataset is not None:
            offline_dataloader = create_dataloader(offline_dataset)
            offline_dataloader = accelerator.prepare(offline_dataloader)
            offline_dl_iter = cycle(offline_dataloader)
        
        # Only create online dataloader if dataset has data
        if online_dataset.num_frames > 0:
            online_dataloader = create_dataloader(online_dataset)
            online_dataloader = accelerator.prepare(online_dataloader)
            online_dl_iter = cycle(online_dataloader)
        else:
            if is_main_process:
                logging.warning("Online dataset is still empty, skipping dataloader creation")

        # Check if we have enough episodes to start training
        # Count total episodes: offline + online
        total_episodes = (
            (offline_dataset.num_episodes if offline_dataset is not None else 0)
            + online_dataset.num_episodes
        )
        if total_episodes < cfg.online.min_episodes_for_training:
            if is_main_process:
                logging.info(
                    f"Only {total_episodes} total episodes (offline: "
                    f"{offline_dataset.num_episodes if offline_dataset is not None else 0}, "
                    f"online: {online_dataset.num_episodes}), "
                    f"need {cfg.online.min_episodes_for_training}. Skipping training phase."
                )
            continue

        # Phase 3: Train on both offline and online datasets
        # Each training step consists of two sub-steps:
        # 1. Train on offline dataset with cmp=False
        # 2. Train on online dataset with cmp=True
        if is_main_process:
            offline_info = (
                f"{offline_dataset.num_episodes} episodes from offline dataset"
                if offline_dataset is not None
                else "no offline dataset"
            )
            online_info = f"{online_dataset.num_episodes} episodes from online dataset"
            logging.info(
                f"Training for {cfg.online.train_steps_per_iteration} steps "
                f"({offline_info}, {online_info})"
            )

        policy.train()
        iteration_start_step = step
        for train_step in range(cfg.online.train_steps_per_iteration):
            # Step 1: Train on offline dataset with cmp=False (if available)
            if offline_dataset is not None and offline_dataset.num_episodes > 0:
                try:
                    start_time = time.perf_counter()
                    offline_batch = next(offline_dl_iter)
                    offline_batch = preprocessor(offline_batch)
                    train_tracker.dataloading_s = time.perf_counter() - start_time

                    train_tracker, output_dict = update_policy(
                        train_tracker,
                        policy,
                        offline_batch,
                        optimizer,
                        cfg.optimizer.grad_clip_norm,
                        accelerator=accelerator,
                        lr_scheduler=lr_scheduler,
                        cmp=False,  # Train offline data with cmp=False
                    )

                    step += 1
                    train_tracker.step()
                except StopIteration:
                    # Reset iterator if exhausted
                    offline_dl_iter = cycle(offline_dataloader)
                    offline_batch = next(offline_dl_iter)
                    offline_batch = preprocessor(offline_batch)
                    train_tracker, output_dict = update_policy(
                        train_tracker,
                        policy,
                        offline_batch,
                        optimizer,
                        cfg.optimizer.grad_clip_norm,
                        accelerator=accelerator,
                        lr_scheduler=lr_scheduler,
                        cmp=False,
                    )
                    step += 1
                    train_tracker.step()

            # Step 2: Train on online dataset with cmp=True (if available)
            if (online_dataset.num_episodes > 0 and online_dataloader is not None
                    and online_dataset.num_frames >= cfg.online.min_frames_for_online_training):
                try:
                    start_time = time.perf_counter()
                    online_batch = next(online_dl_iter)
                    online_batch = preprocessor(online_batch)
                    train_tracker.dataloading_s = time.perf_counter() - start_time

                    train_tracker, output_dict = update_policy(
                        train_tracker,
                        policy,
                        online_batch,
                        optimizer,
                        cfg.optimizer.grad_clip_norm,
                        accelerator=accelerator,
                        lr_scheduler=lr_scheduler,
                        cmp=True,  # Train online data with cmp=True
                    )

                    step += 1
                    train_tracker.step()
                except StopIteration:
                    # Reset iterator if exhausted
                    online_dl_iter = cycle(online_dataloader)
                    online_batch = next(online_dl_iter)
                    online_batch = preprocessor(online_batch)
                    train_tracker, output_dict = update_policy(
                        train_tracker,
                        policy,
                        online_batch,
                        optimizer,
                        cfg.optimizer.grad_clip_norm,
                        accelerator=accelerator,
                        lr_scheduler=lr_scheduler,
                        cmp=True,
                    )
                    step += 1
                    train_tracker.step()

            # Log and save checkpoints after each training iteration
            # (which includes both offline and online training steps)
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
            is_saving_step = step % cfg.save_freq == 0

            if is_log_step:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_log_dict["iteration"] = iteration + 1
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step:
                if is_main_process:
                    logging.info(f"Checkpoint policy after step {step}")
                    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=step,
                        cfg=cfg,
                        policy=accelerator.unwrap_model(policy),
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )
                    update_last_checkpoint(checkpoint_dir)
                    if wandb_logger:
                        wandb_logger.log_policy(checkpoint_dir)

                accelerator.wait_for_everyone()

        if is_main_process:
            total_episodes = (
                (offline_dataset.num_episodes if offline_dataset is not None else 0)
                + online_dataset.num_episodes
            )
            total_frames = (
                (offline_dataset.num_frames if offline_dataset is not None else 0)
                + online_dataset.num_frames
            )
            logging.info(
                f"Iteration {iteration + 1} complete: "
                f"collected {cfg.online.collect_episodes_per_iteration} episodes, "
                f"trained {step - iteration_start_step} steps. "
                f"Total: {total_episodes} episodes ({offline_dataset.num_episodes if offline_dataset is not None else 0} offline + {online_dataset.num_episodes} online), "
                f"{total_frames} frames"
            )

    # Final checkpoint
    if cfg.save_checkpoint and is_main_process:
        logging.info(f"Final checkpoint at step {step}")
        checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=step,
            cfg=cfg,
            policy=accelerator.unwrap_model(policy),
            optimizer=optimizer,
            scheduler=lr_scheduler,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        update_last_checkpoint(checkpoint_dir)
        if wandb_logger:
            wandb_logger.log_policy(checkpoint_dir)

    if collect_env_dict:
        close_envs(collect_env_dict)

    if is_main_process:
        logging.info("End of online training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    init_logging()
    online_train_main()


if __name__ == "__main__":
    main()

