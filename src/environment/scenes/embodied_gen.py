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

import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Dict, Union, Tuple
import numpy as np
import sapien
import torch
from collections import defaultdict
from functools import partial

from scipy.spatial.transform import Rotation as R
from pathlib import Path

import alf

from src.environment.scenes.embodied_gen_assets import (
    TaskObjects, load_assets_from_layout_file, load_layout,
)
from src.environment.scenes.utils import (
    get_simple_obj_name, largest_first_dart_throw, sample_pose, to_world_frame,
)
from src.environment.tasks import get_tasks
from src.robots.maniskill_widowx import BridgeWidowX

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import BaseBridgeEnv
from mani_skill.utils import sapien_utils, io_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import (
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)
from tqdm import tqdm
from embodied_gen.models.gs_model import GaussianOperator
from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
from embodied_gen.utils.geometry import quaternion_multiply
from embodied_gen.utils.simulation import (
    SIM_COORD_ALIGN, )

__all__ = ["EmbodiedGenScene"]


C922_RESOLUTION = np.array([640., 480.])
C922_INTRINSICS = np.array([[623.588, 0, 319.501], [0, 623.588, 239.545],
                            [0, 0, 1]])

AGENT_BRIDGE_TABLE_QPOS = np.array([
    -0.01840777,
    0.0398835,
    0.22242722,
    -0.00460194,
    1.36524296,
    0.00153398,
    0.037,
    0.037,
])



def compute_lookat_point(pose: Pose, z_heights: torch.Tensor) -> torch.Tensor:
    """Compute the intersection point of a camera's forward ray with an XY-plane at a specified z-height.
    This function is batched to accept batched camera poses and batched z_heights.

    Args:
        pose: Pose with batched fields
            - pose.p: (B, 3) positions
            - pose.q: (B, 4) quaternions
        z_heights: (B,) tensor of z-heights

    Returns:
        intersections: A (B, 3) tensor containing intersection points.
    """
    eyes = torch.as_tensor(pose.p)  # (B, 3)
    quats = torch.as_tensor(pose.q)  # (B, 4)

    R = quaternion_to_matrix(quats)  # (B, 3, 3)
    forward = R[:, :, 0]  # (B, 3)

    denom = forward[:, 2]  # (B,)

    # (B,)
    t = (z_heights - eyes[:, 2]) / denom
    intersections = eyes + t[:, None] * forward  # (B, 3)

    return intersections


@alf.configurable
@register_env("EmbodiedGenScene", max_episode_steps=None)
class EmbodiedGenScene(BaseDigitalTwinEnv):
    """
    This class adapts EmbodiedGen's pick-and-place environment for ALF.

    It also reuses ManiSkill's BaseDigitalTwinEnv's green screening logic
    to properly portray static backgrounds rendered from a 3D Gaussian Splat model.

    Note that each scene should be stored as a directory task_%04d in a
    main directory specified by an ENV variable EMBODIED_GEN_SCENES.

    Each environment samples a layout independently from the selected scenes.

    For inference, it is recommended to set

        --conf_param _cache_gs3d_models.parallel_load=True

    to load in the gs3d models faster. For training, models should be loaded sequentially.
    """
    default_graspable_objects = ["baked_green_cube_3cm"]
    default_placeable_objects = [
        "bridge_plate_objaverse",
        "table_cloth_generated_shorter",
    ]
    static_placeable_objects = []
    static_placeable_objects_pos = []

    def __init__(
        self,
        *args,
        graspable_objects: List[str],
        placeable_objects: List[str],
        bbox_local: tuple[float, float, float,
                          float] = (0.20, -0.15, 0.40, 0.15),
        obj_sample_orientations: bool = True,
        use_embodied_gen_objects: bool = False,
        randomize_assets: bool = False,
        max_num_scenes: int = None,
        train_scene_ids_file_path: str = None,
        eval_scene_id: int = None,
        num_envs: int = 1,
        camera_resolution_scale: float = 0.5,
        camera_pos_perturb: np.ndarray = None,
        robot_height_perturb: float = None,
        robot_init_jointpos_perturb: float = None,
        lighting_randomization: bool = False,
        **kwargs,
    ):
        """
        Args:
            graspable_objects: List of graspable objects.
            placeable_objects: List of placeable objects.
            bbox_local: Bounding box in the form of (x_min, y_min, x_max, y_max) of the objects
                to spawned w.r.t. to the robot's base frame. Graspable and placeable objects
                will be randomly spawned within the bbox without overlaps.
            obj_sample_orientations: Whether to sample object orientations.
            use_embodied_gen_objects: Whether to use the implicit objects and tasks in each
                EmbodiedGen scene. If False, build tasks from the configured objects.
            randomize_assets: If True, will randomize object instances from an asset library.
                This library should resize at ${EMBODIED_GEN_SCENES}/asset_lib
            max_num_scenes: The maximum number of scenes to use. It will use the first
                max_num_scenes out of all available ones in ${EMBODIED_GEN_SCENES} or the
                ordered set specified by train_scene_ids_file_path. If None, will use all
                available scenes. Note that each scene takes approximately 100MB of RAM.
            train_scene_ids_file_path: A path to a text file containing a list of unique scene IDs to be
                used for training. Useful for creating a random subset of size N from all available scenes.
            eval_scene_id: Set to evaluate on a single scene indexed by its ID. Can also be used to
                train on a single scene as well.
            num_envs: Number of environments.
            camera_resolution_scale: Camera resolution scale.
            camera_pos_perturb: An optional 2x3 array of camera position perturbations.
                If provided, the camera position will be perturbed with uniformly sampled
                +-deltas from the camera_pos_perturb array. The orientation of the camera
                will be computed to look at the same point of the table using the precomputed
                "look at" point. If None, no perturbation will be applied.
                TODO: Add camera look at point perturbations too.
            robot_height_perturb: A height perturb limit for the robot's z pos. Will be sampled
                uniformly from ~ U[0, robot_height_perturb].
            robot_init_jointpos_perturb: A += perturb value for the robot's initial joint angles.
                Will be sampled for every non-gripper joint angle from ~ U[-rijp, +rijp].
            lighting_randomization: Whether to randomize the lighting including hue and brightness.
        """
        # Add small offset in z-axis to avoid collision.
        self._objs_z_offset = 0.025
        self._robot_z_offset = 0.002

        self._use_embodied_gen_objects = use_embodied_gen_objects

        # Detect all available scenes
        scenes_path = os.getenv("EMBODIED_GEN_SCENES")
        if not scenes_path or not Path(scenes_path).is_dir():
            raise ValueError("Set EMBODIED_GEN_SCENES to the directory containing the task_#### scenes.")
        base_dir = Path(scenes_path)
        scene_dirs = [
            p for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith("task_")
        ]

        self._asset_dir = base_dir / "asset_lib"
        self._randomize_assets = randomize_assets
        if randomize_assets:
            assert self._asset_dir.exists(
            ), f"{self._asset_dir} does not exist."

        # If we don't have explicit scene ids, we'll sort the scenes for repeatability.
        if train_scene_ids_file_path is None:
            scene_dirs = sorted(scene_dirs,
                                key=lambda x: int(x.name.split("_")[1]))
            self._scene_dirs = [base_dir / sd for sd in scene_dirs]
        else:
            train_scene_ids = np.loadtxt(train_scene_ids_file_path,
                                         ndmin=1,
                                         dtype=np.int64)
            num_available_scenes = len(scene_dirs)
            assert len(train_scene_ids) <= num_available_scenes
            assert max(train_scene_ids) < num_available_scenes
            self._scene_dirs = [
                base_dir / f"task_{i:04d}" for i in train_scene_ids
            ]

        self._num_scenes = len(self._scene_dirs)
        if max_num_scenes is not None:
            self._num_scenes = min(self._num_scenes, max_num_scenes)
        if self._num_scenes <= 0:
            raise ValueError(f"No EmbodiedGen scenes selected in {base_dir}.")

        self._eval_scene_id = eval_scene_id
        if self._eval_scene_id is not None:
            assert train_scene_ids_file_path is None
            assert 0 <= self._eval_scene_id < self._num_scenes
            self._num_scenes = 1
            self._scene_dirs = [self._scene_dirs[self._eval_scene_id]]

        # Parse the selected layouts and their task objects once.
        self._scene_infos: List[LayoutInfo] = []
        self._scene_tasks: list[TaskObjects] = []
        for scene_dir in self._scene_dirs[:self._num_scenes]:
            layout, task = load_layout(scene_dir / "layout.json")
            self._scene_infos.append(layout)
            self._scene_tasks.append(task)

        # Compute the robot's pose from the layout files
        self._robot_poses = self.compute_robot_init_pose()

        self._current_tasks: List[str] = []

        # =============== Necessary attributes for BaseDigitalTwinEnv
        self.model_db = io_utils.load_json(
            Path(__file__).resolve().parents[1] / "assets/bridge_object_db.json")
        self.scene_setting = "flat_table"
        self.obj_static_friction = 0.5
        self.obj_dynamic_friction = 0.5
        self.rgb_overlay_paths = {}
        self._rgb_overlay_images = {}
        # =============================================================

        self._objs = {}
        self._obj_sample_orientations = obj_sample_orientations
        self._graspable_objects = graspable_objects
        self._placeable_objects = placeable_objects
        self._fixed_task_ids = None
        if not use_embodied_gen_objects:
            self._fixed_tasks = get_tasks(
                self._graspable_objects,
                [*self._placeable_objects, *self.static_placeable_objects])
            if not self._fixed_tasks:
                raise ValueError("Fixed-object scenes require at least one task.")

        # These are only used if use_embodied_gen_objects is True
        # These data containers contain a list of (name, actor) where each entry is per env.
        self._eg_graspable_objects: List[Tuple[str, Actor]] = []
        self._eg_placeable_objects: List[Tuple[str, Actor]] = []
        # We don't care about the names of the table.
        self._eg_tables: List[Actor] = []
        # This data container contain a nested list of (name, actor) where each entry is per env.
        self._eg_distractor_objects: List[List[Tuple[str, Actor]]] = []

        # Compute the bounding box in world frame
        bbox_local = list(bbox_local)
        bbox_wf1 = []
        bbox_wf2 = []
        for robot_pose in self._robot_poses:
            bbox_wf1.append(
                to_world_frame(robot_pose=robot_pose,
                               pos=np.array([*bbox_local[:2], 0]))[0])
            bbox_wf2.append(
                to_world_frame(robot_pose=robot_pose,
                               pos=np.array([*bbox_local[2:], 0]))[0])
        bbox_wf1 = np.array(bbox_wf1)[:, :2]
        bbox_wf2 = np.array(bbox_wf2)[:, :2]
        self._bbox = np.concatenate((bbox_wf1, bbox_wf2), axis=1)

        # This will be initialized in self._setup_scene()
        self._camera_mount = None
        self._camera_pos_perturb = camera_pos_perturb
        if camera_pos_perturb is not None:
            assert camera_pos_perturb.shape == (2, 3)
            self._camera_pos_perturb = torch.tensor(camera_pos_perturb)
        self._camera_resolution_scale = camera_resolution_scale
        self._camera_mount_pose = self._cache_world_camera_poses()
        self._look_at_point = compute_lookat_point(
            self._camera_mount_pose,
            torch.tensor(self._robot_poses[:, 2], dtype=torch.float32))
        self._robot_height_perturb = robot_height_perturb
        self._robot_init_jointpos_perturb = robot_init_jointpos_perturb
        self._lighting_randomization = lighting_randomization

        self._cached_gs3d_envs = []
        self._background_scene_ids = None
        self._background_needs_render = True

        # This will contain the scene id (value) for each environment (index).
        # Can be used to reference the layout file and robot pose.
        self._scene_ids = np.random.randint(low=0,
                                            high=len(self._scene_infos),
                                            size=num_envs,
                                            dtype=np.int64)

        super().__init__(
            *args,
            robot_uids=kwargs.pop("robot_cls"),
            num_envs=num_envs,
            **kwargs,
        )

        self._cache_gs3d_models()
        # Note self._rgb_overlay_images is used by BaseDigitalTwinEnv for green screening.
        self._render_and_cache_static_backgrounds()

    def _cache_world_camera_poses(self):
        """
        Cache the batched camera poses w.r.t. the world frame for each environment.

        Returns:
            A Pose object containing the camera poses w.r.t. the world frame.
        """
        p = BridgeWidowX.TABLE_CAM_POSE_LOCAL.p
        q = BridgeWidowX.TABLE_CAM_POSE_LOCAL.q
        camera_positions = []
        camera_quats = []
        for robot_pose in self._robot_poses:
            pos, quat = to_world_frame(robot_pose=robot_pose, pos=p, quat=q)
            camera_positions.append(pos)
            camera_quats.append(quat)
        return Pose.create_from_pq(p=torch.tensor(camera_positions),
                                   q=torch.tensor(camera_quats))

    @staticmethod
    def _load_one_cpu(scene_dir, scene_info):
        bg_node = scene_info.relation[Scene3DItemEnum.BACKGROUND.value]
        gs_path = os.path.join(scene_dir, scene_info.assets[bg_node],
                               "gs_model.ply")
        # It's important to keep the GS models in CPU especially if we have
        # a lot of them. Otherwise, GPU memory will get saturated quickly.
        raw_gs = GaussianOperator.load_from_ply(gs_path, device='cpu')
        x, y, z, qx, qy, qz, qw = scene_info.position[bg_node]
        qx, qy, qz, qw = quaternion_multiply([qx, qy, qz, qw],
                                             [0.7071, 0.0, 0.0, 0.7071])
        init_pose = torch.tensor([x, y, z, qx, qy, qz, qw])
        gs_model = raw_gs.get_gaussians(instance_pose=init_pose)
        return gs_model

    @alf.configurable
    def _cache_gs3d_models(self, parallel_load: bool = False):
        """
        Cache the 3D Gaussian Splat models as loading them is expensive.
        We want to cache these models as the camera angle changes, we may need
        to rerender different view points.

        Args:
            parallel_load: If True, will load the GS3D models in parallel
                using threads. Can greatly speed up the loading process. Should
                only be used for inference as this can cause memory-related crashes
                during training.
        """
        self._cached_gs3d_envs.clear()
        if parallel_load:
            with ThreadPoolExecutor(max_workers=32) as pool:
                self._cached_gs3d_envs = list(
                    tqdm(pool.map(
                        self._load_one_cpu,
                        self._scene_dirs[:self.num_scenes],
                        self._scene_infos[:self.num_scenes],
                    ),
                         total=self.num_scenes,
                         desc="Caching 3DGS models"))
        else:
            for env_idx, scene_info in enumerate(
                    tqdm(self._scene_infos,
                         desc="Caching 3D Gaussian Splat models")):
                self._cached_gs3d_envs.append(
                    self._load_one_cpu(scene_dir=self._scene_dirs[env_idx],
                                       scene_info=scene_info))

    def _render_gs3d_images(self) -> Dict[str, np.ndarray]:
        """
        Helper function for rendering the static background images of the
        3D Gaussian Splat models given the camera's current pose.

        Returns:
            A dictionary with a key in the form of "camera_name-env{env_idx}" and
            a value of the rendered image of shape (height, width, 3).
        """
        sim_coord_align = (torch.tensor(SIM_COORD_ALIGN).to(torch.float32).to(
            self.device))
        cameras = self.scene.sensors.copy()
        # cameras.update(self.scene.human_render_cameras)

        bg_images = dict()
        for env_idx, scene_id in enumerate(self._scene_ids):
            gs_model = self._cached_gs3d_envs[scene_id]
            gs_model.to("cuda:0")
            for key in cameras:
                camera = cameras[key]
                Ks = camera.camera.get_intrinsic_matrix()  # (n_env, 3, 3)
                c2w = camera.camera.get_model_matrix()  # (n_env, 4, 4)
                result = gs_model.render(
                    c2w[env_idx] @ sim_coord_align,
                    Ks[env_idx],
                    image_width=camera.config.width,
                    image_height=camera.config.height,
                )
                bg_images[f"{key}-env{env_idx}"] = result.rgb[..., ::-1]

            # Make sure to move back the GS models back to CPU after being done with them.
            gs_model.to("cpu")

        return bg_images

    def _render_and_cache_static_backgrounds(self):
        """
        Helper function for generating static background images from
        cached 3D Gaussian Splat models.

        The static backgrounds are cached for efficiency.
        """
        bg_images = self._render_gs3d_images()

        images = [bg_images[f"3rd_view_camera-env{i}"] for i in range(self.num_envs)]
        self._rgb_overlay_images = {
            "3rd_view_camera": torch.as_tensor(np.stack(images), device=self.device)
        }
        self._background_scene_ids = self._scene_ids.copy()
        self._background_needs_render = False

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        # Reset has applied the new camera poses before the first observation.
        if self._background_needs_render and self._cached_gs3d_envs:
            self._render_and_cache_static_backgrounds()
        return super()._get_obs_sensor_data(apply_texture_transforms)

    def compute_robot_init_pose(self) -> np.ndarray:
        robot_pose = []
        for scene_info in self._scene_infos:
            robot_node = scene_info.relation[Scene3DItemEnum.ROBOT.value]
            x, y, z, qx, qy, qz, qw = scene_info.position[robot_node]
            robot_pose.append([x, y, z + self._robot_z_offset, qw, qx, qy, qz])

        return np.array(robot_pose)

    @property
    def _default_sim_config(self):
        # Note this gets overwritten by ManiSkillParallelEnv
        return SimConfig(
            scene_config=SceneConfig(
                solver_position_iterations=30,
                # contact_offset=0.04,
                # rest_offset=0.001,
            ),
            # sim_freq=200,
            control_freq=50,
            gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2**20,
                                              max_rigid_patch_count=2**19),
        )

    @property
    def _default_human_render_camera_configs(self):
        return []

    def _load_agent(self, options: dict):
        self.ground = build_ground(self.scene)
        super()._load_agent(options, sapien.Pose(p=[-10, 0, 10]))

    # Reuse the build_actor_helper function from BaseBridgeEnv for quickly instantiating
    # new objects.
    _build_actor_helper = BaseBridgeEnv._build_actor_helper

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:

        resolution = (C922_RESOLUTION *
                      self._camera_resolution_scale).astype(int)
        intrinsics = C922_INTRINSICS.copy()
        intrinsics[:2] *= self._camera_resolution_scale

        return [
            CameraConfig(uid="3rd_view_camera",
                         pose=sapien.Pose(p=np.zeros(3),
                                          q=np.array([1, 0, 0, 0])),
                         width=resolution[0],
                         height=resolution[1],
                         intrinsic=intrinsics,
                         mount=self._camera_mount)
        ]

    def get_contact_info(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dict of shape-(batch_size,) bool tensors between graspable
        and placeable objects. Also returns between (all objects and the
        table) and (the robot and graspable objects).
        E.g.:
          'cube|plate', 'plate|table',
          'cube|cloth', 'cloth|table',
          'cube|table', 'gripper|cube'

        NOTE: Due to simulation granularity, the contact info may be
        inaccurate. For example, the cube may be knocked away when
        no contact is detected.
        """
        scene = self.scene
        # 1a) Get flat list of Entities across all sub-scenes
        all_actors = scene.get_all_actors()
        # → List[Entity] with names like "scene-0_baked_red_cube_3cm", "scene-1_baked_red_cube_3cm",
        #  …  [maniskill.readthedocs.io](https://maniskill.readthedocs.io/en/latest/user_guide/reference/mani_skill.envs.scene.html)

        # 1b) Group them by the suffix after the first underscore
        grouped = defaultdict(list)
        for ent in all_actors:
            prefix, base = ent.name.split("_", 1)
            idx = int(prefix.split("-")[1])  # extract the sub-scene index

            # Different scenes use different names for the table, e.g., "table" or "desk".
            # There can also be adjectives, e.g., "wooden table".
            # To support general ground truth contact detection, we'll use a common
            # identifier "arena" for the table-like object.
            if any(p in base for p in ["table-", "desk-"]):
                base = "arena"

            grouped[base].append((ent, idx))

        # Now:
        # grouped["baked_red_cube_3cm"] == [(cube_ent0, 0), (cube_ent1, 1)]
        # grouped["bridge_plate_objaverse"] == [(plate_ent0, 0), (plate_ent1, 1)], etc.
        def merge_entities(base_name: str):
            ents, idxs = zip(*grouped[base_name])
            scene_idxs = torch.tensor(idxs, dtype=torch.int64)
            # Create a multi-env Actor view for this object
            return Actor.create_from_entities(list(ents), scene, scene_idxs)

        # 1c) Get all actors for graspable and placeable objects and the table
        graspable_actors = {
            get_simple_obj_name(name): merge_entities(name)
            for name in self._graspable_objects
        }
        placeable_actors = {
            get_simple_obj_name(name): merge_entities(name)
            for name in self._placeable_objects
        }
        table_actor = merge_entities(
            "arena")  # or whatever your flat‐table actor is named

        # 2a) Impulses between each pair (shape: [num_envs, 3])
        impulses = {}
        for grasp_obj_name, grasp_actor in graspable_actors.items():
            for place_obj_name, place_actor in placeable_actors.items():
                key = grasp_obj_name + "|" + place_obj_name
                impulses[key] = scene.get_pairwise_contact_impulses(
                    grasp_actor, place_actor)

        # Impulses between each object and the table
        for actors in [graspable_actors, placeable_actors]:
            for obj_name, actor in actors.items():
                key = obj_name + "|table"
                impulses[key] = scene.get_pairwise_contact_impulses(
                    actor, table_actor)

        # 2b) Convert impulses to contact info
        contact_results = {}
        for key, imp in impulses.items():
            contact_results[key] = imp[:, 2] > 0.0

        # 3a) Gather gripper links from the WidowX articulation in articulation_views
        # Merge WidowX articulations across sub-scenes
        raw_arts = scene.get_all_articulations()
        grouped = defaultdict(list)
        for art in raw_arts:
            prefix, base = art.name.split("_", 1)
            idx = int(prefix.split("-")[1])
            grouped[base].append((art, idx))
        arts, idxs = zip(*grouped["widowx250s_bridgedataset_flat_table"])
        wid_art = Articulation.create_from_physx_articulations(
            list(arts), scene, torch.tensor(idxs))
        gripper_links = [
            link for link in wid_art.links
            if "finger_link" in link.name.lower()
        ]

        # 3b) Checks to see if any link (num_envs, num_links) is touching any
        # graspable object
        for grasp_obj_name, grasp_actor in graspable_actors.items():
            masks = [
                torch.linalg.norm(scene.get_pairwise_contact_impulses(
                    grasp_actor, link),
                                  dim=1) > 0.0 for link in gripper_links
            ]
            on_gripper = torch.stack(masks, dim=1).any(
                dim=1)  # any link touching cube
            key = "gripper|" + grasp_obj_name
            contact_results[key] = on_gripper

        return contact_results

    def _setup_scene(self):
        super()._setup_scene()
        builder = self.scene.create_actor_builder()
        # Place the camera mount with displaced x to avoid spawning in overlap.
        # https://github.com/haosulab/ManiSkill/issues/421
        builder.initial_pose = sapien.Pose(p=[1, 0, 0])
        self._camera_mount = builder.build_kinematic(name="camera_mount")

    @property
    def num_scenes(self) -> int:
        return self._num_scenes

    def _load_scene(self, options: dict):
        if not self._use_embodied_gen_objects:
            self._scene_ids = np.random.randint(
                self.num_scenes, size=self.num_envs, dtype=np.int64)
        for i in range(self.num_envs):
            # We just copy SimplerEnv's setting for the robot
            sapien_utils.set_articulation_render_material(
                self.agent.robot._objs[i], specular=0.9, roughness=0.3)

        if not self._use_embodied_gen_objects:
            obj_names = self._graspable_objects + self._placeable_objects
            for i, name in enumerate(obj_names):
                scale = self.model_db[name].get("scale", 1.0)
                # Assign unique initial z-positions here to prevent spawning overlapping objects.
                # This pose gets overwritten appropriately later.
                # https://github.com/haosulab/ManiSkill/issues/421
                actor = self._build_actor_helper(
                    name,
                    scale=scale,
                    initial_pose=sapien.Pose(p=[0, 0, 1 + i]))
                self._objs[name] = actor

                # Keep configured objects visible in the composited scene.
                self.remove_object_from_greenscreen(actor)
        else:
            self._eg_graspable_objects.clear()
            self._eg_placeable_objects.clear()
            self._eg_tables.clear()
            self._eg_distractor_objects.clear()

        all_objects = []
        for env_id, scene_id in enumerate(self._scene_ids):
            task = self._scene_tasks[scene_id]
            env_actors, distractor_names = load_assets_from_layout_file(
                self.scene,
                scene_info=self._scene_infos[scene_id],
                scene_dir=self._scene_dirs[scene_id],
                task=task,
                z_offset=self._objs_z_offset,
                env_idx=env_id,
                table_only=not self._use_embodied_gen_objects,
                randomize_assets=self._randomize_assets,
                base_asset_dir=self._asset_dir)
            all_objects.extend(env_actors.values())

            # If we're using embodied gen objects, sort all the objects into their categories.
            if self._use_embodied_gen_objects:
                self._eg_graspable_objects.append(
                    (task.graspable_label, env_actors[task.graspable]))
                self._eg_placeable_objects.append(
                    (task.placeable, env_actors[task.placeable]))
                self._eg_tables.append(env_actors[task.table])
                distractor_names_and_objs = []
                for name in distractor_names:
                    distractor_names_and_objs.append((name, env_actors[name]))
                self._eg_distractor_objects.append(distractor_names_and_objs)

        for obj in all_objects:
            self.remove_from_state_dict_registry(obj)
        self.all_objects = Actor.merge(all_objects, name="all_objects")
        self.add_to_state_dict_registry(self.all_objects)

        # Remove the EmbodiedGen objects from the greenscreen.
        self.remove_object_from_greenscreen(self.all_objects)

        # Remove the robot from the greenscreen.
        self.remove_object_from_greenscreen(self.agent.robot)

    def reset(self,
              seed: Union[None, int, list[int]] = None,
              options: Union[None, dict] = None):
        options = dict(options or {})
        options.setdefault("reconfigure", True)
        if options["reconfigure"] and "env_idx" in options:
            env_idx = options["env_idx"]
            assert torch.equal(env_idx, torch.arange(self.num_envs,
                                                     device=env_idx.device)), (
                "Scene reconfiguration requires resetting all environments together.")

        if options.pop("next_task", False):
            if self._use_embodied_gen_objects:
                if not options["reconfigure"]:
                    raise ValueError("Switching EmbodiedGen layouts requires reconfiguration.")
                self._scene_ids = np.random.randint(
                    self.num_scenes, size=self.num_envs, dtype=np.int64)
            else:
                completed = self._fixed_task_successes()
                weights = torch.tensor(
                    [task.weight for task in self._fixed_tasks], device=self.device)
                self._fixed_task_ids = torch.multinomial(
                    (~completed) * weights + 1e-5, num_samples=1).squeeze(1)

        return super().reset(seed, options)

    def _load_lighting(self, options: dict):
        # Base light configuration was obtained from BaseBridgeEnv
        color1 = np.array([2.2, 2.2, 2.2])
        color2 = np.array([0.7, 0.7, 0.7])
        if not self._lighting_randomization:
            self.scene.set_ambient_light([0.3, 0.3, 0.3])
            self.scene.add_directional_light(
                [0, 0, -1],
                color1,
                shadow=self.enable_shadow,
                shadow_scale=5,
                shadow_map_size=2048,
            )
            self.scene.add_directional_light([-1, -0.5, -1], color2)
            self.scene.add_directional_light([1, 1, -1], color2)
        else:
            for subscene in self.scene.sub_scenes:
                subscene.set_ambient_light(
                    np.random.uniform(0, 0.6, size=3).tolist())
                brightness = np.random.uniform(0.5, 1.5)
                color1 *= brightness
                color2 *= brightness

                subscene.add_directional_light(
                    [0, 0, -1],
                    color1,
                    shadow=self.enable_shadow,
                    shadow_scale=5,
                    shadow_map_size=2048,
                )
                subscene.add_directional_light([-1, -0.5, -1], color2)
                subscene.add_directional_light([1, 1, -1], color2)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        with torch.device(self.device):
            bs = env_idx.shape[0]
            reset_objects = options.get("reset_objects", True)
            if reset_objects:

                if self._use_embodied_gen_objects:
                    self._current_tasks = [
                        f"put the {grasp_name} on the {place_name}."
                        for (grasp_name, _), (place_name, _) in zip(
                            self._eg_graspable_objects, self._eg_placeable_objects)
                    ]
                else:
                    if self._fixed_task_ids is None:
                        # Start with the first two thirds of tasks, as in fixed-object training.
                        initial_task_count = max(1, len(self._fixed_tasks) * 2 // 3)
                        self._fixed_task_ids = torch.randint(
                            initial_task_count, (self.num_envs,))
                    self._current_tasks = [
                        self._fixed_tasks[i].instr
                        for i in self._fixed_task_ids.tolist()
                    ]

                # This code path assumes that every scene has the exact same objects.
                if not self._use_embodied_gen_objects:
                    # Here, we use a largest first dart throwing algorithm to get a collision-free
                    # configuration within the bounding box.

                    # First, collect all the collision meshes of the objects
                    names = []
                    meshes = []
                    for name, actor in self._objs.items():
                        names.append(name)
                        meshes.append(
                            actor.get_first_collision_mesh(
                                to_world_frame=False))

                    xyz = defaultdict(list)
                    quat = defaultdict(list)

                    # Next, get a collision-free configuration for each object.
                    # No easy way to batch this algorithm so we just have to do it serially
                    # for each environment.
                    for scene_id in self._scene_ids:
                        pose_sampler_func = partial(
                            sample_pose,
                            bbox=self._bbox[scene_id],
                            height=self._robot_poses[scene_id, 2] + 0.02,
                            sample_yaw=self._obj_sample_orientations)
                        res = largest_first_dart_throw(
                            names=names,
                            object_meshes=meshes,
                            pose_sampler=pose_sampler_func)
                        for name, pose in res.items():
                            xyz[name].append(pose[:3, 3])
                            quat[name].append(
                                R.from_matrix(
                                    pose[:3, :3]).as_quat(scalar_first=True))

                    xyz = dict(xyz)
                    quat = dict(quat)

                    # Convert all poses to batched tensors
                    for name in xyz.keys():
                        xyz[name] = torch.tensor(xyz[name])
                        quat[name] = torch.tensor(quat[name])

                    for i, (name, actor) in enumerate(self._objs.items()):
                        actor.set_pose(
                            Pose.create_from_pq(p=xyz[name], q=quat[name]))

                else:
                    # This code path is used if we're using EmbodiedGen's implicit objects.
                    # In other words, each scene contains different objects / names.
                    for env_id, scene_id in enumerate(self._scene_ids):
                        pose_sampler_func = partial(
                            sample_pose,
                            bbox=self._bbox[scene_id],
                            height=self._robot_poses[scene_id, 2] + 0.02,
                            sample_yaw=self._obj_sample_orientations)

                        obj_names = [self._eg_graspable_objects[env_id][0], self._eg_placeable_objects[env_id][0]] + \
                                    [obj_name for obj_name, _ in self._eg_distractor_objects[env_id]]

                        obj_actors = [self._eg_graspable_objects[env_id][1], self._eg_placeable_objects[env_id][1]] + \
                                     [obj_actor for _, obj_actor in self._eg_distractor_objects[env_id]]

                        obj_meshes = [
                            obj_actor.get_first_collision_mesh(
                                to_world_frame=False)
                            for obj_actor in obj_actors
                        ]

                        res = largest_first_dart_throw(
                            names=obj_names,
                            object_meshes=obj_meshes,
                            pose_sampler=pose_sampler_func)

                        # Add z-offset to objects.
                        for name in res.keys():
                            res[name][2, 3] += self._objs_z_offset

                        for name, actor in zip(obj_names, obj_actors):
                            actor.set_pose(
                                Pose.create_from_pq(
                                    p=res[name][:3, 3],
                                    q=R.from_matrix(res[name][:3, :3]).as_quat(
                                        scalar_first=True)))

            # Reset the robot
            robot_poses = self._robot_poses[self._scene_ids].copy()
            if self._robot_height_perturb is not None:
                robot_poses[:, 2] += np.random.uniform(
                    high=self._robot_height_perturb, size=(bs, ))
            self.agent.robot.set_root_pose(robot_poses)

            # No way to individually set different qpos, so we'll just
            # perturb at the episode level.
            init_jointpos = AGENT_BRIDGE_TABLE_QPOS.copy()
            if self._robot_init_jointpos_perturb is not None:
                init_jointpos[:6] += np.random.uniform(
                    low=-self._robot_init_jointpos_perturb,
                    high=self._robot_init_jointpos_perturb,
                    size=6,
                )
            self.agent.reset(init_qpos=init_jointpos)

            # Set the camera pose
            if self._camera_pos_perturb is None:
                self._camera_mount.set_pose(
                    self._camera_mount_pose[self._scene_ids])
            else:
                camera_mount_pos = self._camera_mount_pose.p.clone()
                camera_mount_pos = camera_mount_pos[self._scene_ids]

                # Uniformly sample camera position deltas
                low, high = self._camera_pos_perturb
                low = low.unsqueeze(0)
                high = high.unsqueeze(0)
                perturb = low + (high - low) * torch.rand(bs, 3)
                camera_mount_pos += perturb

                # Compute the camera pose using the original pose's
                # "look at" point on the table.
                all_p = []
                all_q = []
                for env_id, scene_id in enumerate(self._scene_ids):
                    pose = sapien_utils.look_at(
                        eye=camera_mount_pos[env_id],
                        target=self._look_at_point[scene_id])
                    all_p.append(pose.p.squeeze(0))
                    all_q.append(pose.q.squeeze(0))

                all_p = torch.stack(all_p, dim=0)
                all_q = torch.stack(all_q, dim=0)

                pose = Pose.create_from_pq(p=all_p, q=all_q)
                self._camera_mount.set_pose(pose)

            self._background_needs_render = (
                self._camera_pos_perturb is not None
                or not np.array_equal(self._background_scene_ids, self._scene_ids))

    def get_current_tasks(self) -> List[str]:
        """Return the instruction for each environment's current task."""
        return self._current_tasks.copy()

    def env_info_spec(self):
        if self._use_embodied_gen_objects:
            return {
                "scene_id":
                    alf.BoundedTensorSpec(shape=(),
                                          dtype=torch.int64,
                                          minimum=0,
                                          maximum=self.num_scenes - 1)
            }
        else:
            return {}

    def get_env_info(self):
        if self._use_embodied_gen_objects:
            return {
                "scene_id": torch.tensor(
                    self._scene_ids, dtype=torch.int64, device=self.device)
            }
        else:
            return {}

    def _fixed_task_successes(self) -> torch.Tensor:
        contacts = self.get_contact_info()
        return torch.stack(
            [task.succeeded(contacts) for task in self._fixed_tasks], dim=1)

    def get_success_info(self) -> torch.Tensor:
        """Return the raw success mask for the current task in each environment."""
        if not self._use_embodied_gen_objects:
            return self._fixed_task_successes().gather(
                1, self._fixed_task_ids[:, None]).squeeze(1)

        graspable_actors = Actor.merge(
            [obj for _, obj in self._eg_graspable_objects], name="grasp")
        placeable_actors = Actor.merge(
            [obj for _, obj in self._eg_placeable_objects], name="place")
        table_actors = Actor.merge(self._eg_tables, name="table")

        # Compute contact impulses
        impulses = {}
        impulses["grasp|place"] = self.scene.get_pairwise_contact_impulses(
            graspable_actors, placeable_actors)
        impulses["grasp|table"] = self.scene.get_pairwise_contact_impulses(
            graspable_actors, table_actors)
        impulses["place|table"] = self.scene.get_pairwise_contact_impulses(
            placeable_actors, table_actors)

        # Convert impulses to contact info
        contact_results = {}
        for key, imp in impulses.items():
            contact_results[key] = imp[:, 2] > 0.0

        # Gather gripper links from the WidowX articulation in articulation_views
        # Merge WidowX articulations across sub-scenes
        raw_arts = self.scene.get_all_articulations()
        grouped = defaultdict(list)
        for art in raw_arts:
            prefix, base = art.name.split("_", 1)
            idx = int(prefix.split("-")[1])
            grouped[base].append((art, idx))
        arts, idxs = zip(*grouped["widowx250s_bridgedataset_flat_table"])
        wid_art = Articulation.create_from_physx_articulations(
            list(arts), self.scene, torch.tensor(idxs))
        gripper_links = [
            link for link in wid_art.links
            if "finger_link" in link.name.lower()
        ]

        masks = [
            torch.linalg.norm(self.scene.get_pairwise_contact_impulses(
                graspable_actors, link),
                              dim=1) > 0.0 for link in gripper_links
        ]
        on_gripper = torch.stack(masks,
                                 dim=1).any(dim=1)  # any link touching cube
        key = "gripper|grasp"
        contact_results[key] = on_gripper

        success_mask = contact_results["grasp|place"] & ~contact_results[
            "gripper|grasp"] & ~contact_results["grasp|table"]

        return success_mask
