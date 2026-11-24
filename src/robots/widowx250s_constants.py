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
import torch
from alf import BoundedTensorSpec


# Simulated joint limits for the six arm joints and gripper.
WX_CURATED_RANGE = np.array([
      # curated_range # original range (specified in widowx250s_p.xml)
        [-2.0, 2.1],  # waist: "-3.14158 3.14158"
        [-1., 1.8],   # shoulder: "-1.88496 1.98968"
        [-1.0, 1.0],  # elbow: "-2.14675 1.6057"
        [-2.1, 2.1],  # forearm_roll: "-3.14158 3.14158"
        [-1.38, 1.7], # wrist_angle: "-1.74533 2.14675"
        [-2.1, 2.1],  # wrist_rotate: "-3.14158 3.14158"
        [0.0, 1.0],   # gripper: Gripper, 1.0 = fully open, and 0.0 = closed
])  # yapf: disable

# Relaxed shoulder and elbow limits for the WidowX kinematics solver.
WX_TELEOP_RANGE = np.array([
    [-1.5, 1.5],
    [-1.3, 1.8],
    [-1.3, 1.3],
    [-2.1, 2.1],
    [-1.38, 2.1],
    [-2.1, 2.1],
    [0.0, 1.0],
])  # yapf: disable

# Cartesian delta action bounds before the gripper command.
WX_POS_DELTA_LIMIT = 2e-1 * 2
WX_AXIS_ANGLE = 4e-1 * 4  # * 4 for faster teleop rotation
DELTA_SCALE = 0.2
WX_EE_DELTA_RANGE_6 = np.array([
    [-WX_POS_DELTA_LIMIT, WX_POS_DELTA_LIMIT],  # x axis
    [-WX_POS_DELTA_LIMIT, WX_POS_DELTA_LIMIT],  # y axis
    [-WX_POS_DELTA_LIMIT, WX_POS_DELTA_LIMIT],  # z axis
    [-WX_AXIS_ANGLE, WX_AXIS_ANGLE],  # e_x * theta axis_angle
    [-WX_AXIS_ANGLE, WX_AXIS_ANGLE],  # e_y * theta axis_angle
    [-WX_AXIS_ANGLE, WX_AXIS_ANGLE],  # e_z * theta axis_angle
]) * DELTA_SCALE # yapf: disable
WX_EE_DELTA_RANGE = np.concatenate([
    WX_EE_DELTA_RANGE_6,
    np.array([[0.0, 1.0]]),  # gripper: Gripper, 1.0 = fully open, and 0.0 = closed
], axis=0)  # yapf: disable


def widow_action_spec(control_type: str = "joint") -> BoundedTensorSpec:
    """Action spec that is overridden following WX_CURATED_RANGE.
    """
    if control_type == "joint":
        return BoundedTensorSpec(shape=(7, ),
                                 dtype=torch.float32,
                                 minimum=WX_CURATED_RANGE[:, 0],
                                 maximum=WX_CURATED_RANGE[:, 1])
    elif control_type == "delta_ee":
        return BoundedTensorSpec(shape=(7, ),
                                 dtype=torch.float32,
                                 minimum=WX_EE_DELTA_RANGE[:, 0],
                                 maximum=WX_EE_DELTA_RANGE[:, 1])
    else:
        raise ValueError(f"Invalid control type: {control_type}.")
