#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


def vicreg_loss(z1, z2, lambda_param=25.0, mu_param=25.0, nu_param=1.0, gamma=1.0, eps=1e-4):
    """VICReg loss with Variance-Invariance-Covariance Regularization.
    
    Args:
        z1: First representation [batch, num_tokens, dim]
        z2: Second representation [batch, num_tokens, dim]
        lambda_param: Weight for invariance loss (default: 25.0)
        mu_param: Weight for variance loss (default: 25.0)
        nu_param: Weight for covariance loss (default: 1.0)
        gamma: Target standard deviation (default: 1.0)
        eps: Small constant for numerical stability
        
    Returns:
        VICReg loss value [batch, num_tokens]
    """
    batch_size, num_tokens, dim = z1.shape

    # 修复内存泄漏: 计算invariance loss后立即减少维度
    invariance_loss = torch.mean(torch.square(z1 - z2), dim=-1)  # [batch, num_tokens]

    # 修复内存泄漏: 预分配tensor以避免list append和stack
    variance_losses = torch.zeros(num_tokens, device=z1.device, dtype=z1.dtype)
    covariance_losses = torch.zeros(num_tokens, device=z1.device, dtype=z1.dtype)
    
    # 修复内存泄漏: 预计算off_diagonal_mask避免在循环中重复创建
    off_diagonal_mask = 1 - torch.eye(dim, device=z1.device, dtype=z1.dtype)

    for i in range(num_tokens):
        z1_i = z1[:, i, :]
        z2_i = z2[:, i, :]

        # 修复内存泄漏: 合并计算以减少中间tensor
        std_z1 = torch.sqrt(torch.var(z1_i, dim=0) + eps)
        std_z2 = torch.sqrt(torch.var(z2_i, dim=0) + eps)

        var_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))
        variance_losses[i] = var_loss

        # 修复内存泄漏: 直接计算协方差矩阵,避免存储中间变量
        z1_centered = z1_i - torch.mean(z1_i, dim=0, keepdim=True)
        z2_centered = z2_i - torch.mean(z2_i, dim=0, keepdim=True)

        cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)

        # 修复内存泄漏: 合并计算
        cov_loss = (
            torch.sum(torch.square(cov_z1 * off_diagonal_mask)) +
            torch.sum(torch.square(cov_z2 * off_diagonal_mask))
        ) / dim
        covariance_losses[i] = cov_loss

    # 修复内存泄漏: 直接使用tensor而不是stack
    total_loss = (
        lambda_param * invariance_loss +
        mu_param * variance_losses[None, :] +
        nu_param * covariance_losses[None, :]
    )

    return total_loss


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def make_att_2d_masks_with_cls(pad_masks, att_masks, num_cls_prefix=2, num_cls_suffix=2, 
                                 num_action_context=5):
    """Extended attention mask function with special handling for CLS tokens.
    
    This function creates attention masks with special behavior for CLS tokens:
    - Prefix CLS tokens (first num_cls_prefix tokens): can attend to all prefix tokens
    - Suffix CLS tokens (last num_cls_suffix tokens): can attend to first/last n action tokens
    
    Args:
        pad_masks: bool[B, N] - true if valid token, false if padding
        att_masks: int32[B, N] - autoregressive mask (1=causal, 0=bidirectional)
        num_cls_prefix: Number of CLS tokens at the beginning
        num_cls_suffix: Number of CLS tokens at the end
        num_action_context: Number of action tokens the suffix CLS can attend to (from both ends)
    
    Returns:
        att_2d_masks: bool[B, N, N] - 2D attention mask
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # Standard attention mask computation
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    
    # Special handling for suffix CLS tokens
    # The last num_cls_suffix tokens should attend to:
    # 1. First num_action_context action tokens (in the suffix part)
    # 2. Last num_action_context action tokens (just before CLS tokens)
    if num_cls_suffix > 0:
        seq_len = att_2d_masks.shape[-1]
        
        # Allow suffix CLS tokens to attend to everything first
        att_2d_masks[:, -num_cls_suffix:, :] = True
        
        # Then mask out middle action tokens (keep only first/last n)
        # This assumes the action tokens are before the CLS tokens
        # Find where suffix starts (this is approximate, may need adjustment based on actual layout)
        # For now, allow full attention for suffix CLS - can be customized later
        pass
    
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = VLAFlowMatching(config)
        self.reset()
        
        # Apply parameter freezing based on config
        if self.config.train_delta_expert:
            self._freeze_for_delta_expert_training()


    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        # Store previous obs_cls_out for similarity comparison
        # Shape: (batch_size, num_cls_prefix, hidden_dim)
        self._prev_obs_cls_out = None

    def _freeze_for_delta_expert_training(self):
        """Freeze all parameters except delta_expert and its related projections."""
        # Freeze all parameters first
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze delta_expert related parameters
        delta_expert_modules = [
            self.model.vlm_with_expert.delta_expert,
            self.model.delta_action_in_proj,
            self.model.delta_action_out_proj,
            self.model.delta_action_time_mlp_in,
            self.model.delta_action_time_mlp_out,
            self.model.action_context_encoder,
        ]
        
        for module in delta_expert_modules:
            for param in module.parameters():
                param.requires_grad = True
        
        print("✓ Froze all parameters except delta_expert and its projections")

    def _unfreeze_all_parameters(self):
        """Unfreeze all trainable parameters (for normal training)."""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ Unfroze all parameters for normal training")

    def get_optim_params(self) -> dict:
        """Return only trainable parameters."""
        if self.config.train_delta_expert:
            # Return only delta_expert related parameters
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            print(f"✓ Returning {len(trainable_params)} trainable parameters for delta_expert training")
            return trainable_params
        else:
            # Return all parameters (normal training)
            return self.parameters()

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, cal_cache=False,
                         past_key_values=None, prefix_pad_masks=None, initial_x_t=None, initial_time=None,
                         action_context: Tensor | None = None, compute_cls_similarity=False) -> Tensor | tuple:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        result = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
            cal_cache=cal_cache, past_key_values=past_key_values, prefix_pad_masks=prefix_pad_masks,
            initial_x_t=initial_x_t, initial_time=initial_time, action_context=action_context,
            prev_obs_cls_out=self._prev_obs_cls_out, compute_cls_similarity=compute_cls_similarity
        )
        
        # If cal_cache=True, result is (past_key_values, prefix_pad_masks, delta_actions)
        if cal_cache:
            return result  # Return tuple directly
        
        # If compute_cls_similarity=True, result is (actions, obs_cls_out, should_cache)
        if compute_cls_similarity and self.config.use_cls_head:
            actions, obs_cls_out, should_cache = result
            
            # Update stored obs_cls_out for next round
            self._prev_obs_cls_out = obs_cls_out.detach()
            
            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            return actions, obs_cls_out, should_cache
        
        # Otherwise result is just actions
        actions = result

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, cal_cache: bool = False,
                           past_key_values=None, prefix_pad_masks=None, initial_x_t=None, initial_time=None,
                           action_context: Tensor | None = None, compute_cls_similarity: bool = False) -> Tensor | tuple:
        """Predict a chunk of actions.
        
        Args:
            batch: Input batch
            noise: Optional noise
            cal_cache: If True, compute and return KV cache
            past_key_values: Cached KV values
            prefix_pad_masks: Cached prefix masks
            initial_x_t: Initial x_t for continuing denoising
            initial_time: Initial time
            action_context: Action context from queue
            compute_cls_similarity: If True, compute CLS similarity and return should_cache flag
            
        Returns:
            If cal_cache=True: (past_key_values, prefix_pad_masks, delta_actions)
            If compute_cls_similarity=True: (actions, obs_cls_out, should_cache)
            Otherwise: actions
        """
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        result = self._get_action_chunk(
            batch, noise, cal_cache=cal_cache,
            past_key_values=past_key_values, prefix_pad_masks=prefix_pad_masks,
            initial_x_t=initial_x_t, initial_time=initial_time, action_context=action_context,
            compute_cls_similarity=compute_cls_similarity
        )
        return result

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.
        
        根据 config.train_delta_expert 参数选择训练模式：
        - train_delta_expert=True: 只训练 delta_expert
        - train_delta_expert=False: 正常训练主模型
        """
        if self.config.train_delta_expert:
            # 训练 delta_expert 模式
            return self.forward_delta_expert(batch, noise, time)
        else:
            # 正常训练模式
            return self._forward_main_model(batch, noise, time)
    
    def _forward_main_model(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Normal forward pass for main model training."""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        # 修复内存泄漏: 使用 .detach() 而不是 .clone()
        loss_dict["losses_after_forward"] = losses.detach().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.detach().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.detach().mean().item()

        # For backward pass
        loss = losses.mean()
        # For backward pass
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def forward_delta_expert(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Training forward pass for delta_expert.
        
        This method trains the delta_expert network and learnable noise to generate delta actions.
        You can customize this method according to your training objectives.
        
        Args:
            batch: Batch of data containing observations and actions
            noise: Optional noise tensor
            time: Optional time tensor
            
        Returns:
            loss: Training loss
            loss_dict: Dictionary containing detailed loss information
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        
        # Forward pass for delta_expert
        losses = self.model.forward_delta_expert(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        # 修复内存泄漏: 使用 .detach() 而不是 .clone()
        loss_dict["delta_losses_after_forward"] = losses.detach().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["delta_losses_after_in_ep_bound"] = losses.detach().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["delta_losses_after_rm_padding"] = losses.detach().mean().item()

        # For backward pass
        loss = losses.mean()
        loss_dict["delta_loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            load_expert_weights=self.config.load_expert_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        # Delta expert layers
        self.delta_action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.delta_action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)
        self.delta_action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.delta_action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )
        
        # Action context encoder: encodes a sequence of actions (e.g., next 5 actions from queue)
        # Input: (batch_size, num_context_actions, max_action_dim)
        # Output: (batch_size, chunk_size, max_action_dim) - expanded to match chunk_size
        self.action_context_encoder = nn.Sequential(
            nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size),
            nn.ReLU(),
            nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        
        # CLS head mechanism: learnable tokens for contrastive learning
        # pre_cls_param: added at the beginning of prefix (image/language tokens)
        # suf_cls_param: added at the end of suffix (action tokens)
        if self.config.use_cls_head:
            self.pre_cls_param = nn.Parameter(
                torch.randn(1, self.config.num_cls_prefix, self.vlm_with_expert.config.text_config.hidden_size) * 0.02
            )
            self.suf_cls_param = nn.Parameter(
                torch.randn(1, self.config.num_cls_suffix, self.vlm_with_expert.expert_hidden_size) * 0.02
            )
        else:
            # Dummy parameters (not used)
            self.register_buffer('pre_cls_param', torch.zeros(1, 2, self.vlm_with_expert.config.text_config.hidden_size))
            self.register_buffer('suf_cls_param', torch.zeros(1, 2, self.vlm_with_expert.expert_hidden_size))
        
        # MLP projections for CLS tokens (only if CLS head is enabled)
        if self.config.use_cls_head:
            class MLP(nn.Module):
                def __init__(self, in_dim, hidden_dim, out_dim):
                    super().__init__()
                    self.fc1 = nn.Linear(in_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, out_dim)
                    self.activation = nn.ReLU()
                
                def forward(self, x):
                    x = self.activation(self.fc1(x))
                    x = self.fc2(x)
                    return x
            
            vlm_hidden_size = self.vlm_with_expert.config.text_config.hidden_size
            expert_hidden_size = self.vlm_with_expert.expert_hidden_size
            
            self.obs_cls_proj = MLP(vlm_hidden_size, vlm_hidden_size, expert_hidden_size)
            self.act_cls_proj = MLP(expert_hidden_size, vlm_hidden_size, expert_hidden_size)
        else:
            # Dummy modules (not used)
            self.obs_cls_proj = nn.Identity()
            self.act_cls_proj = nn.Identity()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        
        # Concatenate embeddings BEFORE adding CLS tokens
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        
        # Add pre_cls_param tokens at the beginning (if enabled)
        if self.config.use_cls_head:
            pre_cls_tokens = self.pre_cls_param.expand(bsize, -1, -1)
            embs = torch.cat([pre_cls_tokens, embs], dim=1)
            
            # Update masks for CLS tokens
            # CLS tokens can attend to all image/language tokens (att_mask=0)
            cls_att_masks = [0] * self.config.num_cls_prefix
            att_masks = cls_att_masks + att_masks
            
            cls_pad_masks = torch.ones(bsize, self.config.num_cls_prefix, dtype=torch.bool, device=device)
            pad_masks = torch.cat([cls_pad_masks, pad_masks], dim=1)
        
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def _add_noise_to_actions(self, x_t: torch.Tensor, noise_scale: float = 0.01) -> torch.Tensor:
        """Add noise to x_t, only to the non-padded dimensions.
        
        Args:
            x_t: Action tensor (batch_size, chunk_size, max_action_dim)
            noise_scale: Scale of noise to add (default: 0.01, which is ~1/10 of typical action range [-1, 1])
            
        Returns:
            Noisy action tensor with same shape
        """
        # Get original action dimension (non-padded part)
        original_action_dim = self.config.action_feature.shape[0] if hasattr(self.config, 'action_feature') else self.config.max_action_dim
        
        # Create a copy to avoid modifying original
        x_t_noisy = x_t.clone()
        
        # Generate Gaussian noise only for non-padded dimensions
        noise = torch.randn(
            x_t.shape[0], 
            x_t.shape[1], 
            original_action_dim,  # Only for real action dimensions
            device=x_t.device,
            dtype=x_t.dtype
        ) * noise_scale
        
        # Add noise only to non-padded part
        x_t_noisy[:, :, :original_action_dim] = x_t[:, :, :original_action_dim] + noise
        
        # Padded part remains unchanged (zero)
        return x_t_noisy

    def forward_delta_expert(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Training forward pass for delta_expert with VICReg loss.
        
        This computes the loss for training delta_expert with CLS head mechanism.
        The delta_expert learns to generate delta actions and the CLS tokens learn
        contrastive representations.
        
        Args:
            images: Input images
            img_masks: Image masks
            lang_tokens: Language tokens
            lang_masks: Language masks
            state: Robot state
            actions: Ground truth actions (target for delta actions)
            noise: Optional noise (if None, uses learnable_noise)
            time: Optional time (if None, uses fixed timestep)
            
        Returns:
            losses: Per-element losses (batch_size x chunk_size x action_dim) including VICReg loss
        """
        bsize = state.shape[0]
        device = state.device
        
        # Compute prefix embeddings (with CLS tokens) and get outputs
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        # Forward through VLM to get prefix outputs (including CLS tokens)
        (prefix_out, _), past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        
        # Extract prefix CLS tokens (first 2 tokens)
        # 修复内存泄漏: detach obs_cls_out 因为它不需要梯度
        obs_cls_out = prefix_out[:, :2, :].detach()  # [batch, 2, hidden_dim]
        
        # Denoise to get x_t using main expert
        # 修复内存泄漏: 使用 torch.no_grad() 因为这部分只是为了生成x_t,不需要梯度
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        initial_time_val = torch.tensor(1.0, dtype=torch.float32, device=device)

        if noise is None:
            x_t = torch.zeros(bsize, self.config.chunk_size, self.config.max_action_dim, device=device)
        else:
            x_t = noise
        
        # 修复内存泄漏: detach past_key_values 用于denoising循环
        past_key_values_detached = {k: {'key_states': v['key_states'].detach(), 
                                        'value_states': v['value_states'].detach()} 
                                   for k, v in past_key_values.items()}
        
        with torch.no_grad():
            time = initial_time_val
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = self.denoise_step(
                    prefix_pad_masks,
                    past_key_values_detached,
                    x_t,
                    expanded_time,
                )
                x_t += dt * v_t
                time += dt
        
        # 修复内存泄漏: detach x_t 并创建新tensor以启用梯度
        x_t = x_t.detach().requires_grad_(False)
        
        # Fixed timestep for delta expert
        timestep = torch.tensor(0.5, dtype=torch.float32, device=device).expand(bsize)
        
        # Add noise to x_t (only to non-padded dimensions)
        x_t_noisy = self._add_noise_to_actions(x_t, noise_scale=self.config.cls_noise_scale)
        
        # Embed suffix with CLS tokens for delta expert
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_delta(x_t_noisy, timestep)
        
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        
        # Forward through delta expert to get suffix outputs (including CLS tokens)
        outputs_embeds, _ = self.vlm_with_expert.forward_delta(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        
        # Extract action predictions (excluding CLS tokens at the end if enabled)
        if self.config.use_cls_head:
            action_out = suffix_out[:, -(self.config.chunk_size + self.config.num_cls_suffix):-self.config.num_cls_suffix]
        else:
            action_out = suffix_out[:, -self.config.chunk_size:]
        action_out = action_out.to(dtype=torch.float32)
        v_t = self.delta_action_out_proj(action_out)
        
        # Compute denoising loss
        target = x_t - x_t_noisy
        denoising_losses = F.mse_loss(target, v_t, reduction="none")  # [batch, chunk_size, action_dim]
        
        # Compute VICReg loss if CLS head is enabled
        if self.config.use_cls_head:
            # Extract prefix CLS tokens (first num_cls_prefix tokens)
            obs_cls_out = prefix_out[:, :self.config.num_cls_prefix, :]  # [batch, num_cls_prefix, hidden_dim]
            
            # Extract suffix CLS tokens (last num_cls_suffix tokens)
            act_cls_out = suffix_out[:, -self.config.num_cls_suffix:, :]  # [batch, num_cls_suffix, hidden_dim]
            
            # Project CLS tokens through MLPs
            obs_cls_proj = self.obs_cls_proj(obs_cls_out)  # [batch, num_cls_prefix, expert_hidden_dim]
            act_cls_proj = self.act_cls_proj(act_cls_out)  # [batch, num_cls_suffix, expert_hidden_dim]
            
            # Compute VICReg loss
            vicreg = vicreg_loss(
                obs_cls_proj,
                act_cls_proj,
                lambda_param=self.config.vicreg_lambda,
                mu_param=self.config.vicreg_mu,
                nu_param=self.config.vicreg_nu,
            )
            vicreg_mean = torch.mean(vicreg, dim=-1)  # [batch]
            
            # Combine losses: denoising loss + weighted VICReg loss
            # The VICReg loss is scalar per batch, expand it to match denoising_losses shape
            vicreg_loss_expanded = vicreg_mean[:, None, None].expand_as(denoising_losses)
            total_losses = denoising_losses + self.config.vicreg_weight * vicreg_loss_expanded
        else:
            # No VICReg loss, only denoising loss
            total_losses = denoising_losses
        
        return total_losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, 
                      cal_cache=False, past_key_values=None, prefix_pad_masks=None, 
                      initial_x_t=None, initial_time=None, action_context=None,
                      prev_obs_cls_out=None, compute_cls_similarity=False) -> Tensor | tuple:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)
        
        Args:
            images: Image inputs
            img_masks: Image masks
            lang_tokens: Language tokens
            lang_masks: Language masks
            state: State inputs
            noise: Optional noise tensor
            cal_cache: If True, returns (past_key_values, prefix_pad_masks, delta_actions) after computing KV cache
            past_key_values: Cached KV values for continuing from intermediate state
            prefix_pad_masks: Cached prefix padding masks
            initial_x_t: Initial x_t for continuing denoising
            initial_time: Initial time value for continuing denoising
            action_context: Action context from queue (batch_size, num_actions, action_dim)
                          Used to generate delta_actions when cal_cache=True
            prev_obs_cls_out: Previous observation CLS tokens for similarity comparison
                            Shape: (batch_size, num_cls_prefix, hidden_dim)
            compute_cls_similarity: If True, compute and return CLS similarity and cache decision
            
        Returns:
            If cal_cache=True: tuple of (past_key_values, prefix_pad_masks, delta_actions)
            If compute_cls_similarity=True: tuple of (actions, obs_cls_out, should_cache)
            Otherwise: final actions (x_t)
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Variables to store CLS outputs for similarity computation
        obs_cls_out = None
        should_cache = True  # Default: cache KV values
        
        # If we have cached values, skip prefix computation and go straight to denoising
        if past_key_values is None or prefix_pad_masks is None:
            # Normal flow: compute prefix embeddings and KV cache
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            
            # Compute image and language key value cache
            # Get prefix outputs if CLS head is enabled
            if self.config.use_cls_head:
                (prefix_out, _), past_key_values = self.vlm_with_expert.forward(
                    attention_mask=prefix_att_2d_masks,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=self.config.use_cache,
                    fill_kv_cache=True,
                )
                # Extract prefix CLS tokens
                obs_cls_out = prefix_out[:, :self.config.num_cls_prefix, :]  # [batch, num_cls_prefix, hidden_dim]
                
                # Compute similarity with previous obs_cls_out if requested
                if compute_cls_similarity and prev_obs_cls_out is not None:
                    # Use obs_cls_out[0] (first CLS token of current round)
                    # and prev_obs_cls_out[1] (second CLS token of previous round)
                    current_cls = obs_cls_out[:, 0, :]  # [batch, hidden_dim]
                    prev_cls = prev_obs_cls_out[:, 1, :]  # [batch, hidden_dim]
                    
                    # Compute cosine similarity
                    # Normalize vectors
                    current_cls_norm = F.normalize(current_cls, p=2, dim=-1)
                    prev_cls_norm = F.normalize(prev_cls, p=2, dim=-1)
                    
                    # Compute similarity for each sample in batch
                    similarity = torch.sum(current_cls_norm * prev_cls_norm, dim=-1)  # [batch]
                    
                    # Average across batch
                    avg_similarity = torch.mean(similarity)
                    
                    # Decide whether to cache based on similarity threshold
                    # If similarity > threshold, the scene hasn't changed much, so cache
                    # If similarity <= threshold, scene has changed, don't cache (force recompute)
                    should_cache = avg_similarity.item() > self.config.cls_similarity_threshold
            else:
                _, past_key_values = self.vlm_with_expert.forward(
                    attention_mask=prefix_att_2d_masks,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=self.config.use_cache,
                    fill_kv_cache=True,
                )

            # If cal_cache, generate delta_actions and return cache
            if cal_cache:
                delta_actions = self.generate_delta_actions(
                    prefix_pad_masks, past_key_values, bsize, device, action_context
                )
                return past_key_values, prefix_pad_masks, delta_actions

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        initial_time_val = torch.tensor(1.0, dtype=torch.float32, device=device)

        # Otherwise continue with full denoising
        x_t = noise
        time = initial_time_val
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            # Euler step
            x_t += dt * v_t
            time += dt
        
        # Return based on what was requested
        if compute_cls_similarity and self.config.use_cls_head:
            return x_t, obs_cls_out, should_cache
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def embed_suffix_delta(self, noisy_actions, timestep):
        """Embed suffix for delta expert (similar to embed_suffix but uses delta projections)."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using delta MLP
        action_emb = self.delta_action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.delta_action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.delta_action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        
        # Concatenate embeddings BEFORE adding CLS tokens
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        
        # Add suf_cls_param tokens at the end (if enabled)
        if self.config.use_cls_head:
            suf_cls_tokens = self.suf_cls_param.expand(bsize, -1, -1)
            embs = torch.cat([embs, suf_cls_tokens], dim=1)
            
            # Update masks for suffix CLS tokens
            # These CLS tokens should have special attention pattern (described below)
            cls_suf_att_masks = [1] * self.config.num_cls_suffix  # First set to 1 (will customize in make_att_2d_masks_with_cls)
            att_masks = att_masks + cls_suf_att_masks
            
            cls_suf_pad_masks = torch.ones(bsize, self.config.num_cls_suffix, dtype=torch.bool, device=device)
            pad_masks = torch.cat([pad_masks, cls_suf_pad_masks], dim=1)
        
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def denoise_step_delta(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step using delta_expert."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_delta(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward_delta(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.delta_action_out_proj(suffix_out)
        return v_t

    def encode_action_context(self, action_context: torch.Tensor) -> torch.Tensor:
        """Encode action context (e.g., next 5 actions from queue) to delta_expert input.
        
        Args:
            action_context: Action context tensor (batch_size, num_context_actions, action_dim)
            
        Returns:
            Encoded action context (batch_size, chunk_size, max_action_dim)
        """
        batch_size = action_context.shape[0]
        num_context_actions = action_context.shape[1]
        
        # Encode each action in the context
        # (batch_size, num_context_actions, action_dim) -> (batch_size, num_context_actions, action_dim)
        encoded_actions = self.action_context_encoder(action_context)
        
        # Expand to chunk_size by repeating and interpolating
        if num_context_actions < self.config.chunk_size:
            # Repeat to match chunk_size (simple strategy: repeat last action)
            repeat_times = (self.config.chunk_size + num_context_actions - 1) // num_context_actions
            expanded = encoded_actions.repeat(1, repeat_times, 1)[:, :self.config.chunk_size, :]
        elif num_context_actions > self.config.chunk_size:
            # Downsample if we have too many actions
            # Use linear interpolation
            expanded = torch.nn.functional.interpolate(
                encoded_actions.transpose(1, 2),  # (B, action_dim, num_context_actions)
                size=self.config.chunk_size,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (B, chunk_size, action_dim)
        else:
            expanded = encoded_actions
        
        return expanded

    def generate_delta_actions(
        self,
        prefix_pad_masks,
        past_key_values,
        batch_size,
        device,
        action_context: torch.Tensor = None,
    ):
        """Generate delta actions using delta_expert with action context.
        
        Args:
            prefix_pad_masks: Padding masks from prefix
            past_key_values: Cached KV values
            batch_size: Batch size
            device: Device to use
            action_context: Action context from queue (batch_size, num_actions, action_dim)
                          If None, uses zero tensor as fallback
            
        Returns:
            delta_actions: Generated delta actions
        """
        # Encode action context or use zero tensor as fallback
        if action_context is not None:
            x_t = self.encode_action_context(action_context)
        else:
            # Fallback: use zero tensor
            x_t = torch.zeros(
                batch_size, self.config.chunk_size, self.config.max_action_dim,
                dtype=torch.float32, device=device
            )
        
        # Fixed timestep for delta expert (you can make this configurable)
        timestep = torch.tensor(0.5, dtype=torch.float32, device=device).expand(batch_size)
        
        # Apply one denoising step using delta_expert
        v_t = self.denoise_step_delta(
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
        )
        
        return v_t
