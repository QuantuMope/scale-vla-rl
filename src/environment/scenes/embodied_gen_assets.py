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

# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Read EmbodiedGen layouts and instantiate their assets in ManiSkill."""

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import trimesh
from scipy.spatial.transform import Rotation as R

from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
from embodied_gen.utils.geometry import quaternion_multiply
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.actor import Actor


def load_actor_from_urdf(
    scene: sapien.Scene | ManiSkillScene,
    file_path: str,
    object_type: Literal["grasp", "place", "distractor", "table"],
    pose: sapien.Pose | None = None,
    env_idx: int = None,
    use_static: bool = False,
    # update_mass: bool = False,
    max_short_side: float = None,
    max_height: float = None,
    height_filter: float = 0.15,
) -> sapien.pysapien.Entity | None:
    """
    This is a modified version of load_actor_from_urdf from embodied_gen.utils.simulation.
    It adds an auto-scaling feature which ensures the shortest xy side of an object is within
    a certain limit to be graspable. It can also scale the height of an object independently.

    Args:
        scene: The scene we are spawning actors into.
        file_path: The URDF file path of the actor.
        object_type: The object type.
        pose: The initial pose.
        env_idx: The environment index for the actor.
        use_static: Whether the object should be static or not.
        max_short_side: The max length the shortest side (between x and y) of an object can be.
            Will be used to compute a scaling factor.
        max_height: The max height a grasp or place object can be.
            If the height is greater than this limit, an individual height scale will be computed.
            Is only used if max_short_side is not None.
        height_filter: A height filter for filtering out distractor objects that are too tall.
            Certain tall objects (e.g., lamps) cause too much occlusion and clutter.
            Is only used if max_short_side is not None.
    """

    def _get_local_pose(origin_tag: ET.Element | None) -> sapien.Pose:
        local_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        if origin_tag is not None:
            xyz = list(map(float, origin_tag.get("xyz", "0 0 0").split()))
            rpy = list(map(float, origin_tag.get("rpy", "0 0 0").split()))
            qx, qy, qz, qw = R.from_euler("xyz", rpy, degrees=False).as_quat()
            local_pose = sapien.Pose(p=xyz, q=[qw, qx, qy, qz])

        return local_pose

    tree = ET.parse(file_path)
    root = tree.getroot()
    node_name = root.get("name")
    file_dir = os.path.dirname(file_path)
    visual_mesh = root.find(".//visual/geometry/mesh")
    visual_file = visual_mesh.get("filename")
    visual_file = os.path.join(file_dir, visual_file)
    collision_mesh = root.find(".//collision/geometry/mesh")
    collision_file = collision_mesh.get("filename")
    collision_file = os.path.join(file_dir, collision_file)

    xy_scale = 1.0
    height_scale = 1.0
    if max_short_side is not None:
        # Load in the mesh to compute size via bounding box bounds.
        mesh = trimesh.load(visual_file)
        size = mesh.bounds[1, [0, 2]] - mesh.bounds[0, [
            0, 2
        ]]  # exclude z which is dim 1
        shortest = float(size.min())
        if shortest > max_short_side:
            xy_scale = max_short_side / shortest

        height = mesh.bounds[1, 1] - mesh.bounds[0, 1]

        # We want to avoid spawning objects that are too tall and cause occlusion
        # e.g., lamps.
        if object_type == "distractor" and height > height_filter:
            return None

        height_scale = xy_scale
        if max_height is not None:
            if height > max_height and max_height / height < xy_scale:
                height_scale = max_height / height

    visual_scale = visual_mesh.get("scale", "1.0 1.0 1.0")
    visual_scale = np.array([float(x) for x in visual_scale.split()])
    visual_scale *= np.array([xy_scale, height_scale, xy_scale])

    collision_scale = collision_mesh.get("scale", "1.0 1.0 1.0")
    collision_scale = np.array([float(x) for x in collision_scale.split()])
    collision_scale *= np.array([xy_scale, height_scale, xy_scale])

    visual_pose = _get_local_pose(root.find(".//visual/origin"))
    collision_pose = _get_local_pose(root.find(".//collision/origin"))

    # static_fric = root.find(".//collision/gazebo/mu1").text
    # dynamic_fric = root.find(".//collision/gazebo/mu2").text

    # For now, we use very high friction values to prevent slipping caused by
    # uneven contact geoms.
    material = physx.PhysxMaterial(
        # static_friction=np.clip(float(static_fric), 0.1, 0.7),
        # dynamic_friction=np.clip(float(dynamic_fric), 0.1, 0.6),
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.00,
    )
    builder = scene.create_actor_builder()

    # Since we downscale the 3D objects to fit in WidowX250S's grip, we also
    # use a simplified down-sampled collision mesh. These are available for
    # graspable target objects and distractors.
    if object_type in ["grasp", "distractor"]:
        simplified_mesh = Path(collision_file).with_name("collision_file_simplified.obj")
        if not simplified_mesh.is_file():
            raise FileNotFoundError(
                f"Missing {simplified_mesh}. Prepare the scene assets with "
                "python -m src.environment.scenes.simplify_mesh "
                "--base-dir \"$EMBODIED_GEN_SCENES\"."
            )
        collision_file = str(simplified_mesh)

    body_type = "static" if use_static else "dynamic"
    builder.set_physx_body_type(body_type)
    builder.add_multiple_convex_collisions_from_file(
        collision_file,
        material=material,
        scale=collision_scale,
        pose=collision_pose,
    )

    builder.add_visual_from_file(
        visual_file,
        scale=visual_scale,
        pose=visual_pose,
    )
    if pose is None:
        pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])

    builder.set_initial_pose(pose)
    if isinstance(scene, ManiSkillScene) and env_idx is not None:
        builder.set_scene_idxs([env_idx])

    actor = builder.build(
        name=node_name if env_idx is None else f"{node_name}-{env_idx}")

    # TODO: support this function if needed, currently errors at the hasattr check
    # if update_mass and hasattr(actor.components[1], "mass"):
    #     node_mass = float(root.find(".//inertial/mass").get("value")) * scale
    #     actor.components[1].set_mass(node_mass)

    return actor


@dataclass(frozen=True)
class TaskObjects:
    graspable: str
    placeable: str
    table: str
    duplicate_graspable: str | None = None

    @property
    def graspable_label(self) -> str:
        if self.duplicate_graspable is not None:
            return self.graspable.removesuffix("1")
        return self.graspable


def _read_task_objects(layout: LayoutInfo, layout_path: Path) -> TaskObjects:
    """Resolve pick-and-place roles from the layout's ordered object list."""
    manipulated_role = Scene3DItemEnum.MANIPULATED_OBJS.value
    manipulated = layout.relation.get(manipulated_role)
    if (not isinstance(manipulated, list) or len(manipulated) not in (2, 3)
            or any(not isinstance(node, str) for node in manipulated)
            or len(set(manipulated)) != len(manipulated)):
        raise ValueError(f"Invalid manipulated objects {manipulated!r} in {layout_path}.")

    graspable, *duplicates, placeable = manipulated
    duplicate = duplicates[0] if duplicates else None
    # Train on the first copy when a layout contains two numbered grasp objects.
    if duplicate is not None and (not graspable.endswith("1")
                                  or duplicate != graspable.removesuffix("1") + "2"):
        raise ValueError(f"Unexpected duplicate grasp object {duplicate!r} in {layout_path}.")

    table_role = Scene3DItemEnum.CONTEXT.value
    table = layout.relation.get(table_role)
    roles = [(node, manipulated_role) for node in manipulated] + [(table, table_role)]
    for node, role in roles:
        if (not isinstance(node, str) or node not in layout.assets
                or layout.objs_mapping.get(node) != role):
            raise ValueError(f"Missing {role} asset {node!r} in {layout_path}.")

    return TaskObjects(graspable, placeable, table, duplicate)


def load_assets_from_layout_file(
    scene: ManiSkillScene | sapien.Scene,
    scene_info: LayoutInfo,
    scene_dir: Path,
    task: TaskObjects,
    z_offset: float = 0.0,
    init_quat: list[float] = [0, 0, 0, 1],
    env_idx: int = None,
    table_only: bool = False,
    randomize_assets: bool = False,
    base_asset_dir: Path = None,
) -> Tuple[dict[str, Actor], List[str]]:
    """
    This is a modified version of load_assets_from_layout_file from embodied_gen.utils.simulation.
    Load assets from `EmbodiedGen` layout-gen output and create actors in the scene.
    Args:
        scene (sapien.Scene | ManiSkillScene): The SAPIEN or ManiSkill scene to load assets into.
        scene_info: The scene info.
        scene_dir: The scene directory path.
        task: The object roles resolved when the layout was loaded.
        z_offset (float): Offset to apply to the Z-coordinate of non-context objects.
        init_quat (List[float]): Initial quaternion (x, y, z, w) for orientation adjustment.
        env_idx (int): Environment index for multi-environment setup.
        table_only: Load only the layout table for fixed-object tasks.
        randomize_assets: Whether to randomize the assets from an asset library provided in
            base_asset_dir.
        base_asset_dir: The path to the asset library. Must be defined if randomize_assets is True.

    Returns:
        The actors keyed by asset name and the names of the loaded distractors.
    """
    if randomize_assets:
        assert base_asset_dir is not None and base_asset_dir.exists()

    distractors = []
    actors = dict()
    nodes = (task.table,) if table_only else scene_info.assets
    for node in nodes:
        role = scene_info.objs_mapping[node]
        if node == task.duplicate_graspable or role == Scene3DItemEnum.BACKGROUND.value:
            continue
        file_dir = scene_info.assets[node]
        file_name = f"{node.replace(' ', '_')}.urdf"
        urdf_file = os.path.join(scene_dir, file_dir, file_name)

        position = scene_info.position[node].copy()
        is_table = node == task.table
        if not is_table:
            position[2] += z_offset

        is_graspable = node == task.graspable
        is_placeable = node == task.placeable
        is_distractor = role == Scene3DItemEnum.DISTRACTOR_OBJS.value
        is_place_distractor = role == Scene3DItemEnum.DISTRACTOR_PLACE_OBJS.value

        if is_graspable:
            object_type = "grasp"
        elif is_placeable:
            object_type = "place"
        elif is_distractor or is_place_distractor:
            object_type = "distractor"
        else:
            object_type = "table"

        # Combine initial quaternion with object quaternion
        x, y, z, qx, qy, qz, qw = position
        qx, qy, qz, qw = quaternion_multiply([qx, qy, qz, qw], init_quat)

        if randomize_assets:
            if role == Scene3DItemEnum.MANIPULATED_OBJS.value:
                obj_name = node.replace(' ', '_').replace("1", "")
                asset_dir = base_asset_dir / obj_name
                asset_files = list(asset_dir.iterdir())
                num_assets = len(asset_files)
                sampled_asset_id = np.random.randint(low=0, high=num_assets)
                sampled_asset_file = asset_files[
                    sampled_asset_id] / "result" / f"{obj_name}.urdf"
                urdf_file = str(sampled_asset_file)

            if is_table:
                asset_dir = base_asset_dir / "table"
                asset_files = list(asset_dir.iterdir())
                num_assets = len(asset_files)
                orig_mesh = trimesh.load(
                    Path(urdf_file).parent / "mesh" /
                    f"{node.replace(' ', '_')}.obj")
                orig_size = orig_mesh.bounds

                sampled_asset_id = np.random.randint(low=0, high=num_assets)
                sampled_asset_file = asset_files[
                    sampled_asset_id] / f"{asset_files[sampled_asset_id].name}.urdf"
                urdf_file = str(sampled_asset_file)

                col_file_name = None
                for fn in (Path(urdf_file).parent / "mesh").iterdir():
                    if "_collision.obj" in str(fn):
                        col_file_name = fn
                        break
                assert col_file_name is not None

                new_mesh = trimesh.load(
                    Path(urdf_file).parent / "mesh" / col_file_name)
                new_size = new_mesh.bounds

                # We need to displace the table correctly to account for the change in size.
                x -= (new_size[0, 0] - orig_size[0, 0])
                z += (new_size[0, 1] - orig_size[0, 1])

        max_height = None
        if is_graspable or is_distractor:
            max_short_side = 0.045
        elif is_placeable:
            max_short_side = 0.15
            max_height = 0.075
        elif is_place_distractor:
            max_short_side = 0.15
        else:
            max_short_side = None

        actor = load_actor_from_urdf(scene,
                                     file_path=urdf_file,
                                     object_type=object_type,
                                     pose=sapien.Pose(p=[x, y, z],
                                                      q=[qw, qx, qy, qz]),
                                     env_idx=env_idx,
                                     use_static=is_table,
                                     max_short_side=max_short_side,
                                     max_height=max_height)

        if actor is None:
            continue
        actors[node] = actor
        if object_type == "distractor":
            distractors.append(node)

    return actors, distractors


def load_layout(layout_path: Path) -> tuple[LayoutInfo, TaskObjects]:
    """Read a layout and resolve its pick-and-place object roles."""
    with layout_path.open() as layout_file:
        layout = LayoutInfo.from_dict(json.load(layout_file))
    return layout, _read_task_objects(layout, layout_path)
