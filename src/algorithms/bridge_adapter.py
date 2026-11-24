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

import json
from typing import Tuple, Dict

import numpy as np
from transformers import AutoTokenizer
from scipy.spatial.transform import Rotation as R

from src.robots.widowx250s_kinematics import WidowX250sKinematicsSolver

from ren_openpi0.agent.env_adapter.base import BaseEnvAdapter
from ren_openpi0.model.vla.processing import VLAProcessor
from ren_openpi0.agent.env_adapter.simpler import BridgeSimplerAdapter


class ManiSkillBridgeAdapter(BridgeSimplerAdapter):
    """
    An adapter for the simulated WidowX using Bridge statistics.
    This adapter is to be used only for simulated ManiSkill scenes.

    It inherits from the upstream Bridge adapter but overrides the initializer so we can drop
    the TensorFlow dependency that was only needed for JSON I/O.
    """

    def __init__(self,
                 dataset_statistics_path: str,
                 pretrained_model_path: str,
                 tokenizer_padding: str,
                 num_image_tokens: int,
                 image_size: Tuple[int, int],
                 max_seq_len: int,
                 action_normalization_type: str = "bound",
                 proprio_normalization_type: str = "bound"):
        BaseEnvAdapter.__init__(self)
        self.image_size = tuple(image_size)
        self.action_normalization_type = action_normalization_type
        self.proprio_normalization_type = proprio_normalization_type
        assert action_normalization_type in ["bound", "gaussian"]
        assert proprio_normalization_type in ["bound", "gaussian"]

        with open(dataset_statistics_path, "r", encoding="utf-8") as f:
            self.dataset_statistics = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path,
                                                       padding_side="right")
        self.processor = VLAProcessor(self.tokenizer,
                                      num_image_tokens=num_image_tokens,
                                      max_seq_len=max_seq_len,
                                      tokenizer_padding=tokenizer_padding)

        self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        self._kin_solver = WidowX250sKinematicsSolver()

    @staticmethod
    def _normalize_gripper_state(raw_gripper_state) -> np.ndarray:
        # In simulation, when the gripper is fully closed, it outputs a value of 0.015 instead of 0
        # When it is fully open, it outputs a value of 0.037.
        # We normalize it to [0, 1] here.
        return np.clip((raw_gripper_state - 0.015) / (0.037 - 0.015), 0.0, 1.0)

    @staticmethod
    def _reverse_postprocess_gripper(action_gripper: np.ndarray) -> np.ndarray:
        """Map executed gripper commands back to the model action domain.

        ``postprocess_gripper()`` thresholds at ``0.5``, which is not bijective.
        The inverse therefore uses the canonical representatives ``0.0`` for
        closed and ``1.0`` for open.
        """
        return np.asarray(action_gripper, dtype=np.float32)

    def reverse_postprocess(self, actions: np.ndarray) -> np.ndarray:
        """Invert ``postprocess()`` for executed env actions.

        The first ``N-1`` action dimensions are inverted exactly. The gripper
        dimension is mapped back to canonical values in ``{0.0, 1.0}`` because
        ``postprocess_gripper()`` applies a threshold and is therefore not
        exactly invertible for arbitrary pre-threshold inputs.
        """
        if self.action_normalization_type == "bound":
            normalized_actions_except_gripper = self.normalize_bound(
                actions[..., :-1],
                np.array(self.dataset_statistics["action"]["p01"])[:-1],
                np.array(self.dataset_statistics["action"]["p99"])[:-1],
                clip_min=-1,
                clip_max=1,
            )
        else:  # gaussian
            normalized_actions_except_gripper = self.normalize_gaussian(
                actions[..., :-1],
                np.array(self.dataset_statistics["action"]["mean"])[:-1],
                np.array(self.dataset_statistics["action"]["std"])[:-1],
            )

        normalized_gripper = self._reverse_postprocess_gripper(actions[...,
                                                                       -1:])

        return np.concatenate(
            [normalized_actions_except_gripper, normalized_gripper], axis=-1)

    def preprocess_proprio(self, obs: Dict) -> np.ndarray:
        jp = obs["joint_pos"]
        ee_tf = self._kin_solver.batched_fwd_kin(jp[:, :6])
        rm_bridge = ee_tf[:, :3, :3]
        rpy_bridge_converted = R.from_matrix(
            rm_bridge @ self.default_rot.T).as_euler("xyz")

        gripper_opening = self._normalize_gripper_state(
            raw_gripper_state=jp[:, -1:])

        raw_proprio = np.concatenate([
            ee_tf[:, :3, 3],
            rpy_bridge_converted,
            gripper_opening,
        ],
                                     axis=1)
        return raw_proprio
