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

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.envs.configs import EnvConfig

from .constants import (
    DEFAULT_FPS,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class SimClientConfig:
    """Configuration for SimClient (simulation environment client).

    This class defines all configurable parameters for the SimClient,
    including network connection, policy settings, and control behavior.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Environment configuration
    env: EnvConfig = field(metadata={"help": "Environment configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Number of actions to actually use from the received chunk (rest are discarded)
    # For example, if server sends 50 actions but we use 20, the remaining 30 are discarded
    max_actions_to_use: int = field(
        default=50,
        metadata={"help": "Maximum number of actions to use from each chunk (default: 20, rest discarded)"}
    )
    
    # Whether to replace remaining actions with new predictions (True) or keep old ones (False)
    # When set to True, receiving new actions will replace any remaining old actions in the queue
    replace_actions_on_new: bool = field(
        default=True,
        metadata={"help": "Replace remaining actions with new predictions (default: True)"}
    )
    
    # Request new actions every N steps (None to disable periodic requests)
    request_new_every_n_steps: int | None = field(
        default=10,
        metadata={"help": "Request new actions every N steps (None to disable, triggers regardless of queue size)"}
    )

    # Task instruction for the simulation to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the simulation to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # Evaluation configuration
    n_episodes: int = field(default=10, metadata={"help": "Number of episodes to evaluate"})
    
    # Seed for reproducibility (if None, uses random initialization)
    seed: int | None = field(default=1000, metadata={"help": "Random seed for reproducibility"})

    # Video recording configuration
    save_videos: bool = field(default=False, metadata={"help": "Save videos of episodes"})
    max_episodes_rendered: int = field(
        default=10, metadata={"help": "Maximum number of episodes to render as videos"}
    )
    videos_dir: str = field(default="outputs/videos", metadata={"help": "Directory to save videos"})

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.max_actions_to_use <= 0:
            raise ValueError(f"max_actions_to_use must be positive, got {self.max_actions_to_use}")
        
        if self.max_actions_to_use > self.actions_per_chunk:
            raise ValueError(
                f"max_actions_to_use ({self.max_actions_to_use}) cannot be greater than "
                f"actions_per_chunk ({self.actions_per_chunk})"
            )

        if self.n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {self.n_episodes}")

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SimClientConfig":
        """Create a SimClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "n_episodes": self.n_episodes,
            "seed": self.seed,
            "max_actions_to_use": self.max_actions_to_use,
            "replace_actions_on_new": self.replace_actions_on_new,
            "save_videos": self.save_videos,
            "max_episodes_rendered": self.max_episodes_rendered,
            "videos_dir": self.videos_dir,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "aggregate_fn_name": self.aggregate_fn_name,
        }

