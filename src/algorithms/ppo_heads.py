# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Horizon Robotics and Hobot Contributors. All Rights Reserved.
# Modified for Scale VLA RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Literal

import torch
import torch.nn as nn


class VLMProprioPooling(nn.Module):
    """Module for pooling VLM + proprio hidden states from RenOpenPi0.

    For the VLM hidden state, supports both mean pooling and learned attention pooling.
    """

    def __init__(self,
                 pool_type: Literal["mean_pool", "attn_pool"],
                 vlm_dim: int,
                 proprio_dim: int,
                 num_heads: int = 4):
        """
        Args:
            pool_type: The pooling strategy for the VLM hidden state
            vlm_dim: The hidden state dimension of each VLM token
            proprio_dim: The hidden state dimension of proprio
            num_heads: The number of heads for the MultiheadAttention block.
                Unused if pool_type is mean_pool.
        """
        super().__init__()
        self._pool_type = pool_type
        self._vlm_dim = vlm_dim
        self._proprio_dim = proprio_dim

        if pool_type == "attn_pool":
            self._query = nn.Parameter(torch.zeros(1, 1, vlm_dim))
            self._attn = nn.MultiheadAttention(vlm_dim,
                                               num_heads=num_heads,
                                               batch_first=True)

    def pool(self, hidden_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [B, L, vlm_dim]
        vlm_hs = hidden_state["vlm"]
        # [B, proprio_dim]
        proprio_hs = hidden_state["proprio"].squeeze(1)

        if self._pool_type == "attn_pool":
            B = vlm_hs.size(0)
            q = self._query.expand(B, -1, -1)
            # [B, vlm_dim]
            vlm_pool = self._attn(query=q, key=vlm_hs,
                                  value=vlm_hs)[0].squeeze(1)
        else:
            # [B, vlm_dim]
            vlm_pool = vlm_hs.mean(dim=1)

        # [B, vlm_dim+proprio_dim]
        return torch.cat([vlm_pool, proprio_hs], dim=1)


class ValueHead(nn.Module):
    """
    Simple value network that takes in RenOpenPi0's VLM + proprio hidden states as input.
    """

    def __init__(self,
                 pool_type: Literal["mean_pool", "attn_pool"],
                 vlm_dim: int = 2048,
                 proprio_dim: int = 1024,
                 num_heads: int = 4):
        """
        Args:
            pool_type: The pooling strategy for the VLM hidden state
            vlm_dim: The hidden state dimension of each VLM token
            proprio_dim: The hidden state dimension of proprio
            num_heads: The number of heads for the MultiheadAttention block.
                Unused if pool_type is mean_pool.
        """
        super().__init__()
        self._pooler = VLMProprioPooling(pool_type, vlm_dim, proprio_dim,
                                         num_heads)
        total_dim = vlm_dim + proprio_dim
        hidden_dim = total_dim // 3
        self._mlp = nn.Sequential(nn.Linear(total_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, 1))

    def forward(self, hidden_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self._pooler.pool(hidden_state)
        return self._mlp(features)


class LearnableFlowNoise(nn.Module):
    """
    A module for generating learnable flow noise.
    This noise can be artificially injected into a flow matching model's integration
    step so that logprobs can be computed.
    """

    def __init__(self,
                 pool_type: Literal["mean_pool", "attn_pool"],
                 vlm_dim: int = 2048,
                 proprio_dim: int = 1024,
                 time_dim: int = 1024,
                 action_projection_dim: int = 1024,
                 num_heads: int = 4,
                 action_chunk_size: int = 4,
                 action_size: int = 7,
                 log_std_min: float = -2.5,
                 log_std_max: float = -2.):
        """
        Args:
            pool_type: The pooling strategy for the VLM hidden state
            vlm_dim: The hidden state dimension of each VLM token
            proprio_dim: The hidden state dimension of proprio
            num_heads: The number of heads for the MultiheadAttention block.
                Unused if pool_type is mean_pool.
            action_chunk_size: The action chunk dimension size.
            action_size: The action dimension.
            log_std_min: The minimum value of the log standard deviation.
                The default value used in ReinFlow for visual manipulation tasks
                was std_min = 0.08 ≈ exp(-2.5)
            log_std_max: The maximum value of the log standard deviation.
                The default value used in ReinFlow for visual manipulation tasks
                was std_max = 0.14 ≈ exp(-2.0)
        """
        super().__init__()
        self._pooler = VLMProprioPooling(pool_type, vlm_dim, proprio_dim,
                                         num_heads)
        total_dim = vlm_dim + proprio_dim + action_projection_dim + time_dim
        hidden_dim = total_dim // 3

        flat_action_size = action_chunk_size * action_size

        self._action_projection = nn.Sequential(
            nn.Linear(flat_action_size, action_projection_dim), nn.ReLU())

        self._mlp = nn.Sequential(nn.Linear(total_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, flat_action_size))

        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

    def forward(self, hidden_state: Dict[str, torch.Tensor],
                action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # [B, action_chunk_size * action_size]
        action = action.reshape(action.shape[0], -1)
        # [B, action_projection_dim]
        a = self._action_projection(action)
        # [B, vlm_dim + proprio_dim]
        h = self._pooler.pool(hidden_state)
        # [B, action_projection_dim + vlm_dim + proprio_dim + time_dim]
        input_features = torch.cat([a, h, time], dim=1)
        # [B, 1]
        raw_log_std = self._mlp(input_features)
        log_std = torch.tanh(raw_log_std)
        log_std = self._log_std_min + 0.5 * (log_std + 1.0) * (
            self._log_std_max - self._log_std_min)
        std = log_std.exp()
        return std
