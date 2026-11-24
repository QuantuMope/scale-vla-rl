# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Horizon Robotics and Hobot Contributors. All Rights Reserved.
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
from typing import List, Optional, Tuple
from scipy.spatial.transform import Rotation as R
from abc import ABC, abstractmethod

import kincpp


class KinematicsSolver(ABC):
    """
    Abstract class for solving manipulator forward and inverse kinematics.

    Users can implement a solver for a particular arm by inheriting this class
    and specifying the ee_home_config and joint_screw_axes params.
    """

    def __init__(self, ik_type: str = "qp", qp_delta_limit: float = 0.10):
        assert ik_type in ["qp", "newton"]
        self._solver = kincpp.KinematicsSolver(self.ee_home_config,
                                               self.joint_screw_axes,
                                               self.lower_joint_limits,
                                               self.upper_joint_limits)
        self._ik_params = kincpp.IKParams()
        self._ik_params.common_params.iters = 5
        self._ik_params.batched_common_params.iters = 5

        if ik_type == "qp":
            self._ik_params.qp_params.delta_limit = qp_delta_limit
            self._ik_params.qp_params.tolerance = 1e-6
            self._ik_params.solver_type = kincpp.QP
        elif ik_type == "newton":
            self._ik_params.solver_type = kincpp.NEWTON

        self._num_joints = self.joint_screw_axes.shape[1]

    @property
    @abstractmethod
    def ee_home_config(self) -> np.ndarray:
        """
        Getter for the end effector home configuration as a (4, 4) TF matrix.

        Returns:
            The end effector home configuration as a (4, 4) TF matrix.

        """
        pass

    @property
    @abstractmethod
    def joint_screw_axes(self) -> np.ndarray:
        """
        Getter for the joint screw axes.

        Returns:
            The joint screw axes as a (6, J) matrix, where J is the number of joints.

        """
        pass

    @property
    @abstractmethod
    def lower_joint_limits(self) -> np.ndarray:
        """
        Getter for the joint lower limits.

        Returns:
            The joint lower limits as an N size vector, where N is the number of joints.

        """
        pass

    @property
    @abstractmethod
    def upper_joint_limits(self) -> np.ndarray:
        """
        Getter for the joint upper limits.

        Returns:
            The joint upper limits as an N size vector, where N is the number of joints.

        """
        pass

    def fwd_kin(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Kincpp forward kinematics wrapper function.

        Args:
            joint_positions: The current joint angles as a (J, ) vector.

        Returns:
            The current end effector TF as a (4, 4) matrix.
        """
        # We index the first self._num_joints elements in case gripper actions are accidentally included.
        return self._solver.forward_kinematics(
            joint_positions[:self._num_joints])

    def inv_kin(self, desired_ee_tf: np.ndarray,
                joint_position_guess: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Kincpp inverse kinematics wrapper function.

        Args:
            desired_ee_tf: The desired end effector (4, 4) TF matrix.
            joint_position_guess: The (J, ) joint position initial guess for the IK solver.
                If the ee delta is small enough, a safe guess is simply the current
                joint position.

        Returns:
            A tuple consisting of
                1. Whether the IK was successful.
                2. The (J, ) joint angles.
        """
        self._ik_params.common_params.desired_ee_tf = desired_ee_tf
        self._ik_params.common_params.q_guess = (
            joint_position_guess[:self._num_joints])
        return self._solver.inverse_kinematics(self._ik_params)

    def create_cartesian_traj(
        self,
        initial_joint_position: np.ndarray,
        desired_xyz: Optional[np.ndarray] = None,
        desired_rpy: Optional[np.ndarray] = None,
        resolution: int = 40,
    ) -> List[np.ndarray]:
        """
        Generates a Cartesian trajectory based on the desired end-effector position and orientation,
        the initial joint positions, and the desired resolution.

        Args:
            initial_joint_position: The initial joint positions of the arm. This is an
                initial seed point for the Newton solver, so changing can result in different
                converged solutions.
            desired_xyz: The desired end-effector position in Cartesian coordinates.
                If None, the current end-effector position is used.
            desired_rpy: The desired end-effector orientation as Roll-Pitch-Yaw angles.
                If None, the current end-effector orientation is used.
            resolution: The number of steps to interpolate the trajectory.

        Returns:
            A list of joint positions corresponding to the interpolated Cartesian trajectory.
        """
        starting_ee_tf = self.fwd_kin(initial_joint_position)
        curr_xyz = starting_ee_tf[:3, -1]

        if desired_xyz is None:
            delta_xyz = np.zeros(3, dtype=np.float32)
        else:
            delta_xyz = (desired_xyz - curr_xyz) / resolution

        if desired_rpy is None:
            delta_rotation_matrix = np.eye(3, dtype=np.float32)
        else:
            # Calculate the desired rotation matrix using scipy
            desired_rotation_matrix = R.from_euler('xyz',
                                                   desired_rpy).as_matrix()

            # Calculate the current rotation matrix from the initial transformation
            current_rotation_matrix = starting_ee_tf[:3, :3]

            # Calculate the incremental rotation matrix
            delta_rotation_matrix = R.from_matrix(
                desired_rotation_matrix @ np.linalg.inv(
                    current_rotation_matrix)).as_rotvec() / resolution
            delta_rotation_matrix = R.from_rotvec(
                delta_rotation_matrix).as_matrix()

        joint_trajectory: List[np.ndarray] = []
        curr_ee_tf = starting_ee_tf
        guess = initial_joint_position
        for i in range(resolution):
            curr_ee_tf[:3, -1] += delta_xyz
            curr_ee_tf[:3, :3] @= delta_rotation_matrix

            success, joint_positions = self.inv_kin(curr_ee_tf, guess)

            # Deployment requires the gripper position as well, so we append a zero
            joint_trajectory.append(joint_positions)

            # Update the guess for the next target.
            guess = joint_positions

        return joint_trajectory

    def batched_fwd_kin(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Kincpp batched forward kinematics wrapper function.
        Uses OMP C++ threads for efficient parallelization.

        For a batch size of 10 and OMP_NUM_THREADS=8, speedup is ~1.15x
        Profiled running widowx250s_kinematics.py

        Args:
            joint_positions: The current joint angles as a (B, J, ) matrix.

        Returns:
            The current end effector TFs as a (B, 4, 4) matrix.
        """
        assert joint_positions.ndim == 2 and joint_positions.shape[1] >= self._num_joints, \
            "joint_positions should be a (B, J) matrix"
        # We need to convert this to a numpy array because the C++ return type is List[np.ndarray]
        return np.array(
            self._solver.batched_forward_kinematics(
                joint_positions[:, :self._num_joints]))

    def batched_inv_kin(
            self, desired_ee_tfs: np.ndarray,
            joint_position_guess: np.ndarray) -> Tuple[List[bool], np.ndarray]:
        """
        Kincpp batched inverse kinematics wrapper function.
        Uses OMP C++ threads for efficient parallelization.

        For a batch size of 10 and OMP_NUM_THREADS=8, speedup is ~4.4x.
        Profiled running widowx250s_kinematics.py

        Args:
            desired_ee_tfs: The desired end effector TFs as a (B, 4, 4) matrix.
            joint_position_guess: The joint position initial guesses for the IK solver as a
                (B, J,) vector. If the ee delta is small enough, a safe guess is simply the current
                joint position.

        Returns:
            A tuple consisting of
                1. A vector of B bools indicating whether the IK was successful.
                2. The (B, J, ) matrix of joint angles
        """
        assert desired_ee_tfs.ndim == 3 and desired_ee_tfs.shape[1:] == (4, 4), \
            "desired_ee_tfs should be a (B, 4, 4) matrix"
        assert joint_position_guess.ndim == 2 and joint_position_guess.shape[1] >= self._num_joints, \
            "joint_position_guess should be a (B, J) matrix"
        self._ik_params.batched_common_params.desired_ee_tfs = desired_ee_tfs
        self._ik_params.batched_common_params.q_guesses = (
            joint_position_guess[:, :self._num_joints])
        return self._solver.batched_inverse_kinematics(self._ik_params)
