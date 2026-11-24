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

import cv2
import os
import torch
import numpy as np
from typing import Literal, Dict, Callable, Optional

import alf
from alf.data_structures import TimeStep, StepType
from alf.environments.alf_environment import AlfEnvironment

from src.robots.maniskill_widowx import BridgeWidowX
from src.environment.instructions import UTF8Encoder
from src.robots.widowx250s_constants import widow_action_spec

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import gymnasium as gym
from src.environment.scenes.embodied_gen import EmbodiedGenScene


_CAMERA_NAME = "3rd_view_camera"


@alf.configurable
def build_mani_skill_vector_env(
        *,
        batch_size: int,
        scene_type: str,
        robot_cls: Callable,
        obj_sample_orientations: bool,
        sim_freq: int,
        control_freq: int,
        solver_iterations: int,
        render_type: str,
        include_depth_in_obs: bool) -> ManiSkillVectorEnv:
    """Build the GPU simulator for the ALF environment."""
    obs_mode = "rgb"
    if include_depth_in_obs:
        obs_mode += "+depth"
    obs_mode += "+segmentation"

    if scene_type != EmbodiedGenScene.__name__:
        raise ValueError(f"Unknown scene: {scene_type}")

    env = gym.make(
        id=scene_type,
        obs_mode=obs_mode,
        robot_cls=robot_cls,
        num_envs=batch_size,
        sensor_configs={_CAMERA_NAME: {"shader_pack": render_type}},
        sim_config={
            "sim_freq": sim_freq,
            "control_freq": control_freq,
            "scene_config": {"solver_position_iterations": solver_iterations},
        },
        control_mode="delta_ee_target",
        obj_sample_orientations=obj_sample_orientations,
        reward_mode="none",
        sim_backend="cuda:0",
        render_backend="cuda:0",
        enable_shadow=True,
    )

    # Make sure to disable any reset / termination logic.
    # We want to control all of this from the alf side.
    return ManiSkillVectorEnv(env, auto_reset=False, ignore_terminations=True)


@alf.configurable(blacklist=['batch_size'])
class ManiSkillParallelEnv(AlfEnvironment):
    """This is an Alf environment for running parallelized GPU sim via
    ManiSkill. It assumes delta_ee_target as the control mode.

    Actions contain Cartesian deltas and a gripper command in WidowX control units.

    Args:
        batch_size: The number of environments to run in this instance. This
            will be set by create_environment.num_parallel_environments.
        robot_cls: The robot class to use. This can be partialed to specify
            sensor configurations.
        scene_type: The ManiSkill scene to use. Note that most scenes are
            configurable (object types / placements, etc.) and can be modified
            via alf.config().
        obj_sample_orientations: Whether to randomly sample object
            yaw orientations.
        action_chunk_size: Can be configured to modify the action spec so that an action
            chunk dimension C is added to the action spec (B, 7) -> (B, C, 7).
            This effectively modifies the MDP so that the action is now the entire action chunk.
            If set, make sure the algorithm's action_chunk_size is identical and its queue_actions
            is set to False since the environment will expect entire chunks.
        max_episode_steps: Number of ALF steps (action chunks when configured)
            before all environments reset together.
        consecutive_success: Number of consecutive successful steps required
            before awarding a reward of one per successful step.
        sim_freq: The simulation frequency (Hz)
        control_freq: The control frequency (Hz)
        solver_iterations: The number of solver iterations for each sim step.
            Default recommended value from ManiSkill is 15 for manipulation
            and 4-8 for locomotion. This can be reduced for significant
            speedups in exchange for loss of accuracy. Should be kept high
            for sim2real and can be lowered otherwise.
        render_type: The type of rendering to use. Note that ray tracing is
            not supported for GPU parallel sim.
        include_depth_in_obs: Whether to include depth in the observation.
        opencv_display: Whether to render the first environment via opencv.
        opencv_train_display: Whether to render the first environment via opencv
            during training. This is useful for things like offline dataset collection.
        max_num_video_frames: The maximum number of parallel environment frames
            to render from the batch. The render() method will render at most
            this many environments from self.batch_size.
        video_fps: Frames per second for the optional environment-side mp4
            recorder.
        video_title: If provided, record environment render frames to an mp4.
            If the value does not end with ".mp4", ".mp4" will be appended.
            If None, no environment-side video is recorded.
        env_id: The alf env id. Unused since this is a batched environment.
    """
    metadata = {'render.modes': ['rgb_array']}

    def __init__(
            self,
            batch_size: int,
            robot_cls: Callable = BridgeWidowX,
            scene_type: str = "EmbodiedGenScene",
            obj_sample_orientations: bool = False,
            action_chunk_size: int = None,
            sim_freq: int = 200,
            control_freq: int = 5,
            solver_iterations: int = 15,
            render_type: Literal['minimal', 'default'] = 'default',
            include_depth_in_obs: bool = False,
            opencv_display: bool = False,
            opencv_train_display: bool = False,
            max_num_video_frames: int = 20,
            video_fps: int = 5,
            video_title: Optional[str] = None,
            env_id: int = 0,
            max_episode_steps: int = 200,
            consecutive_success: int = 1):
        super().__init__()

        if max_episode_steps < 1 or consecutive_success < 1:
            raise ValueError("Episode length and success streak must be positive.")
        if action_chunk_size is not None and action_chunk_size < 1:
            raise ValueError("Action chunks must contain at least one action.")

        self._max_num_video_frames = max_num_video_frames
        self._video_fps = video_fps
        self._video_writer = None
        self._video_frame_size = None
        self._video_path = None
        if video_title is not None:
            self._video_path = (video_title if video_title.endswith(".mp4")
                                else f"{video_title}.mp4")
            video_dir = os.path.dirname(self._video_path)
            if video_dir:
                os.makedirs(video_dir, exist_ok=True)

        self._env = build_mani_skill_vector_env(
            batch_size=batch_size,
            scene_type=scene_type,
            robot_cls=robot_cls,
            obj_sample_orientations=obj_sample_orientations,
            sim_freq=sim_freq,
            control_freq=control_freq,
            solver_iterations=solver_iterations,
            render_type=render_type,
            include_depth_in_obs=include_depth_in_obs)

        self._num_envs = batch_size
        self._max_episode_steps = max_episode_steps
        self._consecutive_success = consecutive_success
        self._instruction_encoder = UTF8Encoder()

        ######################### Observation Spec #############################

        obs_spec = self._env.single_observation_space
        height, width, channels = obs_spec["sensor_data"][_CAMERA_NAME]["rgb"].shape
        assert channels == 3
        img_shape = (channels, height, width)

        joint_spec = obs_spec["agent"]["qpos"]
        pose_spec = obs_spec["agent"]["controller"]["arm"]["target_pose"]

        self._observation_spec = {
            "rgb":
                alf.BoundedTensorSpec(img_shape,
                                      dtype=torch.uint8,
                                      minimum=0,
                                      maximum=255),
            "joint_pos":
                alf.TensorSpec(joint_spec.shape, dtype=torch.float32),
            "joint_vel":
                alf.TensorSpec(joint_spec.shape, dtype=torch.float32),
            "curr_target_pose":
                alf.TensorSpec(pose_spec.shape, dtype=torch.float32),
            "instrs": self._instruction_encoder.instruction_spec,
        }
        self._include_depth_in_obs = include_depth_in_obs
        if include_depth_in_obs:
            self._observation_spec["depth"] = alf.TensorSpec(
                (1, img_shape[1], img_shape[2]), dtype=torch.float32)

        ######################### Action Spec #############################
        assert self._env.single_action_space.shape == (7, )
        self._action_spec = widow_action_spec("delta_ee")
        if action_chunk_size is not None:
            self._action_spec = alf.BoundedTensorSpec(
                shape=(action_chunk_size, *self._action_spec.shape),
                dtype=self._action_spec.dtype,
                minimum=np.tile(self._action_spec.minimum, (action_chunk_size, 1)),
                maximum=np.tile(self._action_spec.maximum, (action_chunk_size, 1)),
            )

        self._action_chunk_size = action_chunk_size

        ############################# Misc Setup ################################

        self._episode_steps = 0
        self._env_id = torch.arange(batch_size, device=self._env.device,
                                    dtype=torch.int64)

        self._curr_frames = torch.zeros((batch_size, height, width, channels),
                                        dtype=torch.uint8,
                                        device=self._env.device)
        self._opencv_train_display = opencv_train_display
        self._opencv_display = opencv_display or opencv_train_display
        self.render_mode = "rgb_array"

    @property
    def mani_skill_env(self) -> ManiSkillVectorEnv:
        return self._env

    @property
    def is_tensor_based(self):
        return True

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._num_envs

    def seed(self, seed: Optional[int] = None):
        alf.utils.common.set_random_seed(seed)

    def observation_spec(self) -> alf.NestedTensorSpec:
        return self._observation_spec

    def action_spec(self) -> alf.BoundedTensorSpec:
        return self._action_spec

    def env_info_spec(self) -> alf.NestedTensorSpec:
        return {
            **self._env.unwrapped.env_info_spec(),
            "success": alf.TensorSpec((), dtype=torch.float32),
        }

    def _time_step(self, obs, step_type, reward, prev_action) -> TimeStep:
        env_info = self._env.unwrapped.get_env_info()
        # ALF sums env_info over the episode. Report completion once.
        env_info["success"] = (self._episode_success.float()
                               if step_type == StepType.LAST
                               else torch.zeros_like(reward))
        return TimeStep(
            observation=obs,
            step_type=torch.full_like(self._env_id, step_type, dtype=torch.int32),
            reward=reward,
            # The time limit truncates the episode, allowing PPO to bootstrap.
            discount=torch.ones_like(reward),
            prev_action=prev_action,
            env_id=self._env_id,
            env_info=env_info)

    def _convert_obs(self, ms_obs: Dict) -> Dict:
        camera_obs = ms_obs["sensor_data"][_CAMERA_NAME]
        new_obs = {
            "rgb":
                camera_obs["rgb"].permute(0, 3, 1, 2).contiguous(),
            "joint_pos":
                ms_obs["agent"]["qpos"],
            "joint_vel":
                ms_obs["agent"]["qvel"],
            "curr_target_pose":
                ms_obs["agent"]["controller"]["arm"]["target_pose"],
            "instrs": self._instructions,
        }
        if self._include_depth_in_obs:
            # Sapien depth is in int16 with unit of mm. Convert to float32 meters
            new_obs["depth"] = (
                camera_obs["depth"].permute(
                    0, 3, 1, 2).contiguous()).to(torch.float32) / 1000.0

        return new_obs

    def _step(self, action: torch.Tensor) -> TimeStep:
        if self._episode_steps == self._max_episode_steps:
            return self._reset()

        capture_substeps = self._action_chunk_size is not None and (
            self._video_path is not None or self._opencv_display)
        # In case AMP is enabled, make sure to disable autocast for
        # ManiSkill's batch stepping, otherwise torch dtype conflicts can occur.
        with torch.amp.autocast("cuda", enabled=False):
            prev_action = action.float()
            action = alf.utils.spec_utils.clip_to_spec(
                prev_action, self._action_spec)

            actions = ((action,) if self._action_chunk_size is None
                       else action.unbind(dim=1))
            for substep_action in actions:
                ms_obs = self._env.step(substep_action)[0]
                if capture_substeps:
                    self._curr_frames.copy_(ms_obs["sensor_data"][_CAMERA_NAME]["rgb"])
                    if self._video_path is not None:
                        self._record_video_frame()
                    else:
                        self.render()

        obs = self._convert_obs(ms_obs)
        if not capture_substeps:
            self._curr_frames.copy_(ms_obs["sensor_data"][_CAMERA_NAME]["rgb"])
            if self._video_path is not None:
                self._record_video_frame()
            elif self._opencv_train_display:
                self.render()

        raw_success = self._env.unwrapped.get_success_info()
        self._success_streak = torch.where(
            raw_success, self._success_streak + 1, 0)
        success = self._success_streak >= self._consecutive_success
        self._episode_success |= success
        self._episode_steps += 1
        timed_out = self._episode_steps == self._max_episode_steps
        step_type = StepType.LAST if timed_out else StepType.MID

        return self._time_step(obs, step_type, success.float(), prev_action)

    def _reset(self) -> TimeStep:
        ms_obs, _ = self._env.reset(options={
            "next_task": self._episode_steps == self._max_episode_steps
        })
        tasks = self._env.unwrapped.get_current_tasks()
        self._instructions = torch.as_tensor(
            self._instruction_encoder.encode(tasks), device=self._env.device)
        self._episode_steps = 0
        self._success_streak = torch.zeros_like(self._env_id)
        self._episode_success = torch.zeros_like(self._env_id, dtype=torch.bool)
        obs = self._convert_obs(ms_obs)
        self._curr_frames.copy_(ms_obs["sensor_data"][_CAMERA_NAME]["rgb"])
        prev_action = torch.zeros(
            (self._num_envs, *self._action_spec.shape),
            dtype=self._action_spec.dtype, device=self._env.device)
        return self._time_step(obs, StepType.FIRST,
                               torch.zeros_like(self._env_id, dtype=torch.float32),
                               prev_action)

    def _record_video_frame(self) -> None:
        frame = self.render(mode="rgb_array")
        if self._video_writer is None:
            height, width = frame.shape[:2]
            self._video_frame_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(self._video_path, fourcc,
                                                 self._video_fps,
                                                 self._video_frame_size)
            if not self._video_writer.isOpened():
                raise RuntimeError(
                    f"Failed to open video writer for {self._video_path}")

        if frame.shape[1::-1] != self._video_frame_size:
            frame = cv2.resize(frame,
                               self._video_frame_size,
                               interpolation=cv2.INTER_AREA)
        self._video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        if hasattr(super(), "close"):
            super().close()

    def render(self, mode: str = "rgb_array"):
        """
        This function will at most render the first ``self._max_num_video_frames`` envs.
        """
        num_frames = min(self.batch_size, self._max_num_video_frames)
        base_np = self._curr_frames[:num_frames].cpu().numpy()
        curr_frames_np = list(base_np)
        if len(curr_frames_np) >= 6:
            max_num_columns = 6
            num_rows = int(np.ceil(len(curr_frames_np) / max_num_columns))
            num_columns = int(np.ceil(len(curr_frames_np) / num_rows))
            num_padding_frames = num_rows * num_columns - len(curr_frames_np)
            if num_padding_frames > 0:
                blank = np.zeros_like(curr_frames_np[-1])
                curr_frames_np.extend(
                    [blank.copy() for _ in range(num_padding_frames)])

            # Stack images vertically for 6+ envs, adding rows as needed to
            # keep the rendered image from becoming too wide.
            curr_frames_np = [
                np.vstack(curr_frames_np[i:i + num_rows])
                for i in range(0, len(curr_frames_np), num_rows)
            ]

        # Stack images horizontally
        image = np.hstack(curr_frames_np)

        if self._opencv_display:
            cv2.imshow(f"First {num_frames} envs frames",
                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return image

def load(env_name: str = "",
         batch_size: int = 1,
         env_id: int = 0,
         batched: bool = True,
         **kwargs) -> ManiSkillParallelEnv:
    """Load the environment.
    """
    return ManiSkillParallelEnv(batch_size=batch_size,
                                  env_id=env_id,
                                  **kwargs)


# Need to explicitly set this. getattr doesn't work for default args. Could
# be improved...
load.batched = True
