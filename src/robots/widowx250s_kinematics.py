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

from src.robots.widowx250s_constants import WX_CURATED_RANGE, WX_TELEOP_RANGE
from src.robots.manipulator_kinematics import KinematicsSolver


# TODO: can nest an instance of this class in WidowX250sArm later if needed.
class WidowX250sBaseKinematicsSolver(KinematicsSolver):
    # All the following constants can be found in here:
    # https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/wx250s.html#drawings-and-cad-files

    def __init__(self,
                 elongated_gripper: bool = False,
                 ik_type: str = 'qp',
                 qp_delta_limit: float = 0.10):
        self._x_delta = 0.05 if elongated_gripper else 0.0
        super().__init__(ik_type=ik_type, qp_delta_limit=qp_delta_limit)

    @property
    def ee_home_config(self) -> np.ndarray:
        return np.asarray([[1.0, 0.0, 0.0, 0.458325 + self._x_delta],
                           [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.36065],
                           [0.0, 0.0, 0.0, 1.0]])

    @property
    def joint_screw_axes(self) -> np.ndarray:
        return np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, -0.11065, 0.0, 0.0],
                           [0.0, 1.0, 0.0, -0.36065, 0.0, 0.04975],
                           [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0],
                           [0.0, 1.0, 0.0, -0.36065, 0.0, 0.29975],
                           [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0]]).T

    @property
    def lower_joint_limits(self) -> np.ndarray:
        return WX_CURATED_RANGE[:6, 0]

    @property
    def upper_joint_limits(self) -> np.ndarray:
        return WX_CURATED_RANGE[:6, 1]


class WidowX250sKinematicsSolver(WidowX250sBaseKinematicsSolver):
    """
    Similar to WidowX250sBaseKinematicsSolver, but with more relaxed limits for
    the shoulder and elbow joints. The waist joint is tighter.
    """

    def __init__(self,
                 elongated_gripper: bool = False,
                 ik_type: str = 'qp',
                 qp_delta_limit: float = 0.10):
        super().__init__(elongated_gripper=elongated_gripper,
                         ik_type=ik_type,
                         qp_delta_limit=qp_delta_limit)

    @property
    def lower_joint_limits(self) -> np.ndarray:
        return WX_TELEOP_RANGE[:6, 0]

    @property
    def upper_joint_limits(self) -> np.ndarray:
        return WX_TELEOP_RANGE[:6, 1]


if __name__ == '__main__':
    """
    This script is primarily for profiling kincpp's performance.

    Can choose number of OMP threads via
    OMP_NUM_THREADS=8 python src/robots/widowx250s_kinematics.py
    """
    import time
    from scipy.spatial.transform import Rotation as R
    solver = WidowX250sKinematicsSolver()

    target_ee_tf = np.asarray([[1.0, 0.0, 0.0, 0.458325], [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.36065], [0.0, 0.0, 0.0, 1.0]])
    # 1 cm translation in all directions
    target_ee_tf[:3, 3] -= 0.01

    # 5 degree rotation in pitch and yaw
    ori_delta = 5
    delta_rot = R.from_euler('zy', [ori_delta, ori_delta], degrees=True)
    target_ee_tf[:3, :3] = delta_rot.as_matrix() @ target_ee_tf[:3, :3]
    bs = 10
    batched_target_ee_tf = np.tile(target_ee_tf.reshape((1, 4, 4)), (bs, 1, 1))

    trials = 10000

    # Test FK times
    all_zeros = np.zeros((6, ))
    s = time.perf_counter()
    for _ in range(trials):
        solver.fwd_kin(all_zeros)
    t_fk = time.perf_counter() - s
    print(f"Avg FK time: {t_fk / trials}")

    batched_all_zeros = np.zeros((
        bs,
        6,
    ))
    s = time.perf_counter()
    for _ in range(trials // bs):
        solver.batched_fwd_kin(batched_all_zeros)
    t_b_fk = time.perf_counter() - s
    print(f"Avg batched FK time: {t_b_fk / trials}")

    # Test IK times
    s = time.perf_counter()
    for _ in range(trials):
        solver.inv_kin(target_ee_tf, all_zeros)
    t_ik = time.perf_counter() - s
    print(f"Avg IK time: {t_ik / trials}")

    s = time.perf_counter()
    for _ in range(trials // bs):
        solver.batched_inv_kin(batched_target_ee_tf, batched_all_zeros)
    t_b_ik = time.perf_counter() - s
    print(f"Avg batched IK time: {t_b_ik / trials}")

    print("")

    print(f"Batched FK speedup: {t_fk / t_b_fk}")
    print(f"Batched IK speedup: {t_ik / t_b_ik}")
