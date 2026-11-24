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
import trimesh
from typing import List, Dict, Callable, Tuple, Literal
from loguru import logger
from scipy.spatial.transform import Rotation as R

import alf

import sapien
from mani_skill.utils.structs.pose import Pose as ManiSkillPose

# The Widow's base is not aligned with the world frame origin.
# This makes specifying object positions quite annoying so we'll
# create some helper functions to switch between the two frames.
ROBOT_GLOBAL_POSE: Dict[str, np.ndarray] = {}
# The robot is rotated 180 degrees along world Z.
ROBOT_GLOBAL_QUAT = [0, 0, 0, 1]
ROBOT_GLOBAL_POS_TABLE_SCENE = [0.147, 0.028, 0.870]
ROBOT_GLOBAL_POS_SINK_SCENE = [0.127, 0.060, 0.850]

robot_global_pose_tmp = np.eye(4)
rot_mat = R.from_quat(ROBOT_GLOBAL_QUAT, scalar_first=True).as_matrix()
robot_global_pose_tmp[:3, :3] = rot_mat

# Set the robot global pose for the table scene
robot_global_pose_tmp[:3, 3] = ROBOT_GLOBAL_POS_TABLE_SCENE
ROBOT_GLOBAL_POSE["flat_table"] = robot_global_pose_tmp.copy()

# Set the robot global pose for the sink scene
robot_global_pose_tmp[:3, 3] = ROBOT_GLOBAL_POS_SINK_SCENE
ROBOT_GLOBAL_POSE["sink"] = robot_global_pose_tmp.copy()


def convert_camera_to_world_frame(scene_setting: Literal["flat_table",
                                                         "sink"],
                                  pose_local: sapien.Pose) -> ManiSkillPose:
    """
    Helper function for obtaining a camera's pose in the world frame
    given its pose w.r.t. to the robot's local frame.

    Args:
        scene_setting: The scene setting of the env.
        pose_local: The camera's pose w.r.t. to the robot's local frame.

    Returns:
        The camera's pose in the world frame.
    """
    cam_pos, cam_quat = to_world_frame(scene_setting,
                                       pos=pose_local.p,
                                       quat=pose_local.q)
    return ManiSkillPose.create_from_pq(p=torch.tensor(cam_pos),
                                        q=torch.tensor(cam_quat))


def to_world_frame(
    scene_setting: Literal["flat_table", "sink"] = None,
    robot_pose: np.ndarray = None,
    pos: np.ndarray = np.zeros((3, )),
    quat: np.ndarray = np.array([1, 0, 0, 0]),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function for converting an object in the robot's local base frame
    to the world frame. Either scene_setting (for SimplerEnv scenes) or an explicit
    robot_pose can be provided (but not both). The robot pose in the world
    frame for the provided scene setting must be defined in ROBOT_GLOBAL_POSE.

    Arg:
        scene_setting: The scene setting of the env.
        robot_pose: The robot pose w.r.t. to the world frame. Should be a 7D vector
            with scalar-first quaternion.
        pos: The position w.r.t. to the robot's local base frame.
        quat: The scalar-first quaternion  w.r.t. to the robot's local base
            frame.

    Returns:
        A tuple containing:
            1) The pos in the world frame
            1) The scalar-first quaternion in the world frame
    """
    assert (scene_setting is not None) ^ (robot_pose is not None), \
        "Only one of scene_setting or robot_pose should be defined."

    if scene_setting is not None:
        assert scene_setting in ROBOT_GLOBAL_POSE
        robot_pose = ROBOT_GLOBAL_POSE[scene_setting]
    else:
        # Convert robot_poses to a TF matrix
        tmp = np.eye(4)
        tmp[:3, 3] = robot_pose[:3]
        tmp[:3, :3] = R.from_quat(robot_pose[3:7],
                                  scalar_first=True).as_matrix()
        robot_pose = tmp

    pose = np.eye(4)
    pose[:3, 3] = pos
    pose[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    new_pose = robot_pose @ pose
    new_pos = new_pose[:3, 3]
    new_quat = R.from_matrix(new_pose[:3, :3]).as_quat(scalar_first=True)
    return new_pos, new_quat


def to_robot_frame(
    scene_setting: Literal["flat_table", "sink"],
    pos: np.ndarray = np.zeros((3, )),
    quat: np.ndarray = np.array([1, 0, 0, 0])
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function for converting an object in the world frame to the robot's
    local base frame. The robot pose in the world frame for the provided scene
    setting must be defined in ROBOT_GLOBAL_POSE.

    Arg:
        scene_setting: The scene setting of the env.
        pos: The position w.r.t. to the world frame.
        quat: The scalar-first quaternion  w.r.t. to world frame.

    Returns:
        A tuple containing:
            1) The pos in the robot's local base frame
            1) The scalar-first quaternion in the robot's local base frame
    """
    assert scene_setting in ROBOT_GLOBAL_POSE
    pose = np.eye(4)
    pose[:3, 3] = pos
    pose[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    new_pose = np.linalg.inv(ROBOT_GLOBAL_POSE[scene_setting]) @ pose
    new_pos = new_pose[:3, 3]
    new_quat = R.from_matrix(new_pose[:3, :3]).as_quat(scalar_first=True)
    return new_pos, new_quat


def sample_random_2d_displacements(n: int, r: float) -> torch.Tensor:
    """
    Batched function for uniformly sampling 2D displacements with radius r.

    Args:
        n: batch size
        r: radius.

    Returns:
        A (n, 2) tensor of displacements.
    """
    u = torch.rand(n)  # Uniform[0, 1)
    radii = r * torch.sqrt(u)  # sqrt ensures uniform density
    angles = torch.rand(n) * 2 * torch.pi  # Uniform[0, 2pi)
    displacements = torch.stack(
        (radii * torch.cos(angles), radii * torch.sin(angles)), dim=1)
    return displacements


def sample_random_yaw_quaternions(n: int) -> torch.Tensor:
    """
    Batched function for sampling random yaw quaternions.

    Args:
        n: batch size

    Returns:
        A (n, 4) tensor of scalar-first quaternions.
    """
    angles = torch.rand(n) * 2 * torch.pi  # Uniform[0, 2pi)
    half_angles = angles / 2

    w = torch.cos(half_angles)
    x = torch.zeros(n)
    y = torch.zeros(n)
    z = torch.sin(half_angles)

    quaternions = torch.stack((w, x, y, z), dim=1)  # shape (n, 4)
    return quaternions


def sample_pose(bbox: Tuple[float, float, float, float], height: float,
                sample_yaw: bool) -> np.ndarray:
    """
    Return a random 4x4 pose with randomized yaw and translation.

    Args:
        bbox: The xy bounding box to sample from in the form of (x_min,
            y_min, x_max, y_max)
        height: The static height to use.
        sample_yaw: Whether to sample a random yaw.
    """
    tx = np.random.uniform(bbox[0], bbox[2])
    ty = np.random.uniform(bbox[1], bbox[3])
    pose = np.eye(4)
    pose[:3, 3] = [tx, ty, height]

    if sample_yaw:
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return pose


def get_xy_obb(mesh: trimesh.Trimesh, transform: np.ndarray) -> np.ndarray:
    """Get a 2D oriented bounding box (OBB) from a mesh.

    Args:
        mesh: The collision mesh.
        transform: The pose to move the mesh to.

    Returns:
        A (4, 2) array of OBB vertices.

    """
    box = mesh.bounding_box_oriented.copy()
    box.apply_transform(transform)
    # Project 3D bounding box to 2D
    verts_xy = box.vertices[:, :2]

    # Find the 2D oriented bounding box (4 corners)
    to_origin, extents = trimesh.bounds.oriented_bounds_2D(verts_xy)

    # Half-sizes
    w, h = extents / 2.0

    # Local corners
    local_corners = np.array([
        [-w, -h],
        [w, -h],
        [w, h],
        [-w, h],
    ])

    # Transform corners back
    corners = trimesh.transform_points(local_corners, np.linalg.inv(to_origin))
    return corners


def obb_overlap(corners1: np.ndarray, corners2: np.ndarray) -> bool:
    """
    Check overlap between two oriented bounding boxes (each given by 4 (x,y) corners).

    Args:
        corners1: The first OBB corners.
        corners2: The second OBB corners.

    Returns:
        True if the OBBs overlap, False otherwise.
    """

    def get_axes(corners):
        # Box edges → perpendicular normals
        axes = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            edge = p2 - p1
            axis = np.array([-edge[1], edge[0]])
            axis /= np.linalg.norm(axis)
            axes.append(axis)
        return axes

    def project(corners, axis):
        dots = corners @ axis
        return np.min(dots), np.max(dots)

    # Test all 4 axes from both boxes (8 total, but 4 unique because opposite edges share axes)
    for axis in get_axes(corners1)[:2] + get_axes(corners2)[:2]:
        min1, max1 = project(corners1, axis)
        min2, max2 = project(corners2, axis)
        if max1 < min2 or max2 < min1:
            return False  # separating axis found
    return True


@alf.configurable(whitelist=["verbose", "hard_kill_limit"])
def largest_first_dart_throw(
        names: List[str],
        object_meshes: List[trimesh.Trimesh],
        pose_sampler: Callable[..., np.ndarray],
        max_trials: int = 10,
        verbose: bool = True,
        hard_kill_limit: int = 2000) -> Dict[str, np.ndarray]:
    """
    A largest object first dart throwing algorithm for finding collision-free
    configurations of objects. Note that this algorithm can run indefinitely
    if given too many objects and / or not enough sampling area.

    Collision checks are performed using 2D OOBs and are performed incrementally.
    I.e., for each object, check if it collides with any pre-placed objects. If not,
    we place it and continue. If we can't find a solution after max_trials, we will
    reset and start over from the first object.

    Args:
        names: List of object names.
        object_meshes: List of collision meshes.
        pose_sampler: A function that samples a pose.
        max_trials: The number of trials for an incremental placement before starting over.
        verbose: Whether to print warning messages for scenes taking more than 50 iterations
            to find a collision-free layout. These messages will be printed every 25 iterations.
        hard_kill_limit: A limit to the number of placement attempts before raising a RuntimeError.
            Warning messages will be displayed if half of hard_kill_limit iters is reached regardless
            of whether verbose is set to False.

    Returns:
        A dict of object names to poses where each object is placed collision-free.
    """
    # Before placement, we sort by object area
    objects = list(zip(names, object_meshes))
    objects.sort(key=lambda x: np.prod(x[1].bounding_box.extents[:2]),
                 reverse=True)

    attempt_counter = 0
    while True:
        pose_res = {}
        obbs = []
        success = True

        for name, obj_mesh in objects:
            placed = False
            for _ in range(max_trials):
                pose = pose_sampler()

                # Compute bounding box in XY
                bbox2d = get_xy_obb(obj_mesh, pose)

                # Check overlap with already placed objects
                if any(obb_overlap(bbox2d, other) for other in obbs):
                    continue

                # Passed → accept placement
                pose_res[name] = pose
                obbs.append(bbox2d)
                placed = True
                break

            # After max_trials, we'll reset and keep trying
            if not placed:
                success = False
                break

        if success:
            return pose_res

        attempt_counter += 1

        if verbose and attempt_counter > 50 and attempt_counter % 25 == 0:
            logger.warning(f"Over {attempt_counter+1} placement retries.")

        if attempt_counter > (
                hard_kill_limit //
                2) and attempt_counter % (hard_kill_limit // 10) == 0:
            logger.warning(
                f"Over half of the hard_kill_limit {hard_kill_limit} has been reached."
            )

        if attempt_counter > hard_kill_limit:
            raise RuntimeError(
                f"A placement took more than {hard_kill_limit} iterations.")


SIMPLE_OBJ_NAMES = {
    "green_cube_3cm": "green_cube",
    "yellow_cube_3cm": "yellow_cube",
    "baked_red_cube_3cm": "red_cube",
    "baked_green_cube_3cm": "green_cube",
    "baked_yellow_cube_3cm": "yellow_cube",
    "bridge_spoon_generated_modified": "spoon",
    "bridge_carrot_generated_modified": "carrot",
    "table_cloth_generated_shorter": "cloth",
    "bridge_plate_objaverse": "plate",
    "dummy_sink_target_plane": "drying_rack"
}


def get_simple_obj_name(obj_name: str) -> str:
    return SIMPLE_OBJ_NAMES.get(obj_name, obj_name)
