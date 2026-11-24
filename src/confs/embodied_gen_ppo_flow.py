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

from src.algorithms.bridge_adapter import ManiSkillBridgeAdapter
from src.algorithms.ppo_flow import RenOpenPi0PPOFlowAlgorithm
from src.environment.maniskill_env import load
from src.environment.scenes.embodied_gen import EmbodiedGenScene

import alf
from alf.utils.per_process_context import PerProcessContext
from functools import partial

ddps = PerProcessContext().num_processes
num_para_envs = alf.define_config("num_para_envs", 24 * ddps)
unroll_length = alf.define_config("unroll_length", 100)
num_episodes_per_env = alf.define_config("num_episodes_per_env", 4)
learning_rate = alf.define_config("learning_rate", 2e-5)
domain_randomize = alf.define_config("domain_randomize", True)
randomize_camera_pose = alf.define_config("randomize_camera_pose", "easy")
torch_compile = alf.define_config("torch_compile", False)
action_chunk_size = alf.define_config("action_chunk_size", 4)
num_integration_steps = alf.define_config("num_integration_steps", 1)
train_mode = alf.define_config("train_mode", "lora_full_action_head")
importance_ratio_clip = alf.define_config("importance_ratio_clip", 0.2)
scale_log_prob = alf.define_config("scale_log_prob", 0.2)
log_std_min = alf.define_config("log_std_min", -2.5)
log_std_max = alf.define_config("log_std_max", -2.0)
use_embodied_gen_objects = alf.define_config("use_embodied_gen_objects", True)
max_num_scenes = alf.define_config("max_num_scenes", None)
DEBUG = alf.define_config("DEBUG", False)

assert train_mode in [
    "full_action_head", "lora_action_head", "lora_full", "lora_lm_action_head",
    "lora_full_action_head"
]
assert randomize_camera_pose in ["easy", "hard"]

if train_mode in ["lora_full", "lora_lm_action_head", "lora_full_action_head"]:
    batch_size = 12
    utd_ratio = 1
    num_grad_steps = 10
else:
    batch_size = 300
    utd_ratio = 16
    num_grad_steps = 1

if DEBUG:
    num_para_envs = 2
    batch_size = 2
    unroll_length = 4
    utd_ratio = 1
    num_episodes_per_env = 1
    num_grad_steps = 1

total_steps_per_gpu = int(num_para_envs / ddps * unroll_length)
assert total_steps_per_gpu % batch_size == 0
grad_acc = total_steps_per_gpu // batch_size // num_grad_steps
assert (total_steps_per_gpu // batch_size) % num_grad_steps == 0
batch_size *= ddps
# Each episode occupies one FIRST step plus its action steps.
max_episode_steps = unroll_length // num_episodes_per_env - 1

scene_type = EmbodiedGenScene.__name__
if domain_randomize:
    import numpy as np
    if randomize_camera_pose == "easy":
        camera_pos_perturb = np.array([[-0.05, -0.05, -0.05],
                                       [0.05, 0.05, 0.05]])
    else:
        camera_pos_perturb = np.array([[-0.10, -0.20, -0.15],
                                       [0.10, 0.025, 0.10]])
    alf.config(scene_type,
               robot_init_jointpos_perturb=0.10,
               robot_height_perturb=0.05,
               lighting_randomization=True,
               camera_pos_perturb=camera_pos_perturb)

# Note that we do not use the baked version so that lighting affects the cube
alf.config(scene_type,
           max_num_scenes=max_num_scenes,
           use_embodied_gen_objects=use_embodied_gen_objects,
           graspable_objects=["green_cube_3cm"],
           placeable_objects=["yellow_cube_3cm"])

alf.config("ManiSkillParallelEnv",
           obj_sample_orientations=True,
           action_chunk_size=action_chunk_size,
           max_episode_steps=max_episode_steps,
           consecutive_success=max(1, 3 // action_chunk_size),
           scene_type=scene_type)

alf.config('create_environment',
           env_name=scene_type,
           num_parallel_environments=num_para_envs,
           env_load_fn=load)

# Reduce log bloating when many iterations are used to create a random scene layout.
alf.config("largest_first_dart_throw", verbose=False)

# Ignore loading base model weights
alf.config('Checkpointer.load', strict=False)

# Should be set to False, otherwise there will be a huge spike in GPU memory
# during whole replay buffer training as we cache extremely large tensors for
# action head only finetuning.
alf.config("ReplayBuffer.gather_all", convert_to_default_device=False)

alg_ctor = partial(RenOpenPi0PPOFlowAlgorithm,
                   env_adapter_ctor=ManiSkillBridgeAdapter,
                   torch_compile=torch_compile,
                   action_chunk_size=action_chunk_size,
                   num_integration_steps=num_integration_steps,
                   train_mode=train_mode,
                   scale_log_prob=scale_log_prob,
                   importance_ratio_clip=importance_ratio_clip,
                   log_std_min=log_std_min,
                   log_std_max=log_std_max,
                   optimizer=alf.optimizers.AdamW8bit(
                       lr=learning_rate,
                       grad_accumulation_steps=grad_acc,
                       gradient_clipping=0.5,
                       clip_by_global_norm=True,
                       ignore_param_not_requiring_grad=True))

alf.config("TrainerConfig",
           algorithm_ctor=alg_ctor,
           unroll_length=unroll_length,
           mini_batch_length=1,
           mini_batch_size=batch_size,
           num_updates_per_train_iter=utd_ratio,
           evaluate=False,
           empty_cache=True,
           debug_summaries=True,
           summary_interval=1,
           summarize_grads_and_vars=True,
           summarize_action_distributions=False,
           whole_replay_buffer_training=True,
           clear_replay_buffer=True,
           clear_replay_buffer_but_keep_one_step=False,
           save_checkpoint_upon_crash=not DEBUG,
           num_iterations=1000,
           num_checkpoints=1)
