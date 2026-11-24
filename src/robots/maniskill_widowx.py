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

import numpy as np
import sapien

from src.robots.widowx250s_constants import WX_TELEOP_RANGE

from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    WidowX250SBridgeDatasetFlatTable, )
from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseControllerConfig


class BridgeWidowX(WidowX250SBridgeDatasetFlatTable):
    JOINT_LOWER = WX_TELEOP_RANGE[:6, 0]
    JOINT_UPPER = WX_TELEOP_RANGE[:6, 1]

    TABLE_CAM_POSE_LOCAL = sapien.Pose(
        [0.00, -0.16, 0.36],
        [0.8992917, -0.09263245, 0.35892478, 0.23209205],
    )

    SINK_CAM_POSE_LOCAL = sapien.Pose([-0.00300001, -0.21, 0.39],
                                      [-0.907313, 0.0782, -0.36434, -0.194741])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _sensor_configs(self):
        return []

    @property
    def _controller_configs(self):

        # NOTE: setting interpolate=True for any control mode
        # will cause GPU stepping to break

        arm_common_kwargs = dict(
            joint_names=self.arm_joint_names,
            lower=self.JOINT_LOWER,
            upper=self.JOINT_UPPER,
            stiffness=np.array(self.arm_stiffness) * 2.0,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=False,
            normalize_action=False,
        )
        arm_pd_joint_pos_control = PDJointPosControllerConfig(
            **arm_common_kwargs)

        arm_common_kwargs = dict(
            joint_names=self.arm_joint_names,
            pos_lower=
            -1.0,  # dummy limit, which is unused since normalize_action=False
            pos_upper=1.0,
            rot_lower=-np.pi / 2,
            rot_upper=np.pi / 2,
            stiffness=np.array(self.arm_stiffness) * 2.0,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link="ee_gripper_link",
            urdf_path=self.urdf_path,
            use_delta=True,
            normalize_action=False,
        )

        arm_pd_ee_delta_target_control = PDEEPoseControllerConfig(
            **arm_common_kwargs, use_target=True)

        arm_pd_ee_delta_control = PDEEPoseControllerConfig(**arm_common_kwargs,
                                                           use_target=False)

        gripper_pd_joint_pos_control = PDJointPosMimicControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=-0.04,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
            mimic={
                self.gripper_joint_names[0]: {
                    "joint": self.gripper_joint_names[1]
                }
            })

        joint_controller = dict(arm=arm_pd_joint_pos_control,
                                gripper=gripper_pd_joint_pos_control)
        delta_ee_target_controller = dict(arm=arm_pd_ee_delta_target_control,
                                          gripper=gripper_pd_joint_pos_control)
        delta_ee_controller = dict(arm=arm_pd_ee_delta_control,
                                   gripper=gripper_pd_joint_pos_control)

        controller_configs = dict(joint=joint_controller,
                                  delta_ee_target=delta_ee_target_controller,
                                  delta_ee=delta_ee_controller)

        return controller_configs
