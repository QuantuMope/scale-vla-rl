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

import os
from typing import Dict, NamedTuple, List, Literal, Callable
import torch

import alf
from alf.utils import dist_utils
from alf.algorithms.config import TrainerConfig
from alf.algorithms.ppo_algorithm import PPOAlgorithm, PPOInfo
from alf.algorithms.ppo_loss import PPOLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import AlgStep, TimeStep
from alf.utils.dist_utils import DiagMultivariateNormal
from alf.utils.per_process_context import PerProcessContext

from src.algorithms.bridge_adapter import ManiSkillBridgeAdapter
from src.algorithms.ppo_heads import ValueHead, LearnableFlowNoise
from src.algorithms.openpi0_policy import RenOpenPi0Algorithm, get_train_param_config

from ren_openpi0.model.kv_cache import KVCache


def _setup_torch_compile_cache_dirs_for_ddp():
    """Give each DDP rank its own TorchInductor and Triton caches."""
    ddp_rank = PerProcessContext().ddp_rank
    if ddp_rank < 0:
        return

    cache_vars = ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
    cache_dirs = [os.environ.get(var) for var in cache_vars]
    assert all(cache_dirs), (
        "TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR env vars must be set")

    for var, base_dir in zip(cache_vars, cache_dirs):
        unique_dir = os.path.join(base_dir, f"rank_{ddp_rank}")
        os.makedirs(unique_dir, exist_ok=True)
        os.environ[var] = unique_dir


class RenOpenPi0PPOFlowInfo(NamedTuple):
    # Cached action-related model inputs
    action_inputs: Dict[str, torch.Tensor]

    # All cached intermediate actions necessary for computing the denoising logprob trajectory
    intermediate_actions: torch.Tensor
    ppo: PPOInfo

    # ============== These are used if training only the action_head ==========
    # Cached hidden states for language, image, and proprio from Paligemma
    hidden_states: Dict[str, torch.Tensor] = dict()
    # Cached KV values for language, image, and proprio from Paligemma
    vlm_proprio_kv_caches: Dict[str, List[torch.Tensor]] = dict()

    # ======= These are used if training the full model minus the vision tower ==========
    vision_embeds: torch.Tensor = ()

    # ============== These are used if training the full model ==========
    # Cached model inputs for language, image, and proprio from Paligemma
    vlm_proprio_inputs: Dict[str, torch.Tensor] = dict()


@alf.configurable
class RenOpenPi0PPOFlowAlgorithm(PPOAlgorithm):
    """
    This is a PPO-version of the ReinFlow paper for the RenOpenPi0 model.
    https://arxiv.org/abs/2505.22094v1

    Currently, it finetunes the action head (300M params) while freezing the rest of the
    ren_openpi0 model.
    """

    def __init__(
            self,
            observation_spec: alf.NestedTensorSpec,
            action_spec: alf.NestedTensorSpec,
            reward_spec: alf.NestedTensorSpec,
            env_adapter_ctor: Callable[..., ManiSkillBridgeAdapter],
            action_chunk_size: int = 4,
            num_integration_steps: int = 4,
            importance_ratio_clip: float = 0.2,
            scale_log_prob: float = 1.0,
            action_head_dtype: torch.dtype = torch.bfloat16,
            log_std_min: float = -2.5,
            log_std_max: float = -2.0,
            entropy_regularization: float = None,
            torch_compile: bool = False,
            pool_type: Literal["mean_pool", "attn_pool"] = "attn_pool",
            train_mode: Literal["full_action_head", "lora_action_head",
                                "lora_vlm", "lora_full",
                                "lora_full_action_head"] = "full_action_head",
            lora_rank: int = 32,
            use_grad_checkpointing: bool = False,
            env=None,
            config: 'TrainerConfig' = None,
            optimizer=None,
            debug_summaries: bool = False,
            name: str = "RenOpenPi0PPOFlowAlgorithm"):
        """
        Args:
            observation_spec: observation spec of the environment.
            action_spec: action spec of the environment.
            reward_spec: reward spec of the environment.
            env_adapter_ctor: The constructor for the environment adapter.
            action_chunk_size: The action chunk dimension, C.
            # TODO: Add a separate num_trainable_integration_steps parameter
            num_integration_steps: The number of integration steps, K.
                Note that the number of integration steps will cause the training
                time to linearly increase.
            importance_ratio_clip: The importance ratio clipping ratio.
            scale_log_prob: Scale factor for the log probabilities. Helps to
                maintain numerical stability. Note that this distorts the
                importance ratio by r^scale_log_prob.
            action_head_dtype: The dtype to use for the RenOpenPi0 action head.
                Should be either bfloat16 or float32.
            log_std_min: The min log std for the learnable noise.
            log_std_max: The max log std for the learnable noise.
            entropy_regularization: Coefficient for entropy regularization loss term.
            torch_compile: Whether to torch compile the base RenOpenPi0 model.
                Roughly doubles inference speed for a small upfront cost.
            pool_type: The pooling strategy to use for the VLM hidden states.
            train_mode: Specifies what weights will be allowed to be updated during training.
                "full_action_head": Fully finetune only the action head.
                "lora_action_head": LoRA finetune only the action head.
                "lora_lm_action_head": LoRA finetune Gemma and action head. Freezes SigLip.
                "lora_full": LoRA finetune the entire model.
                "lora_full_action_head" LoRA finetune the VLM and fully finetune the action head.

                Note that if using either full_action_head or lora_action_head modes, the
                PaliGemma forward will be cached during rollout to be used during the train step.

                If using "lora_lm_action_head" mode, the vision embeddings from SigLip will
                be cached.

            lora_rank: The rank of the LoRA matrices. Unused if train_mode is "full_action_head".
            use_grad_checkpointing: Whether to enable grad checkpointing.
            env: ALF env for training the algorithm
            config: TrainerConfig
            optimizer: Optimizer
            debug_summaries: Whether to add debug summaries
            name: Name of the algorithm
        """
        if torch_compile:
            _setup_torch_compile_cache_dirs_for_ddp()

        if "lora" not in train_mode:
            lora_rank = None

        freeze_modules, lora_modules = get_train_param_config(train_mode)

        # Loading in ren_openpi0 creates a large temporary spike in GPU VRAM, so we make sure
        # to do it first here.
        openpi0 = RenOpenPi0Algorithm(
            observation_spec=observation_spec,
            action_spec=action_spec,
            env_adapter_ctor=env_adapter_ctor,
            action_chunk_size=action_chunk_size,
            num_integration_steps=num_integration_steps,
            encoder_dtype=torch.bfloat16,
            action_head_dtype=action_head_dtype,
            mode="train",
            lora_rank=lora_rank,
            lora_target_modules=lora_modules,
            freeze_modules=freeze_modules,
            torch_compile=torch_compile,
            use_grad_checkpointing=use_grad_checkpointing,
        )
        RLAlgorithm.__init__(self,
                             observation_spec=observation_spec,
                             action_spec=action_spec,
                             reward_spec=reward_spec,
                             train_state_spec=(),
                             env=env,
                             config=config,
                             is_on_policy=False,
                             optimizer=optimizer,
                             debug_summaries=debug_summaries,
                             name=name)

        self._openpi0 = openpi0
        self._train_mode = train_mode
        self._action_chunk_size = action_chunk_size
        self._num_integration_steps = num_integration_steps
        self._delta_t = 1.0 / self._num_integration_steps
        # ReinFlow paper suggested using \epsilon = 0.001 for visual manipulation tasks, but test show using
        # the default value of \epsilon = 0.2 is stable and outperforms.
        self._loss = PPOLoss(td_loss_weight=0.5,
                             importance_ratio_clipping=importance_ratio_clip,
                             entropy_regularization=entropy_regularization,
                             debug_summaries=debug_summaries)
        self._entropy_regularization = entropy_regularization
        self._value_head = ValueHead(pool_type=pool_type)
        self._noise_head = LearnableFlowNoise(
            pool_type=pool_type,
            action_chunk_size=action_chunk_size,
            log_std_min=log_std_min,
            log_std_max=log_std_max)
        self._scale_log_prob = scale_log_prob

        assert action_head_dtype in [torch.bfloat16, torch.float32], \
            "action_head_dtype must be torch.bfloat16 or torch.float32"
        self._action_head_dtype = action_head_dtype

        # Precompute the time embeddings since the number of integration steps won't change.
        t = torch.zeros(1, dtype=self._action_head_dtype)
        self._time_cond: List[torch.Tensor] = []
        for _ in range(self._num_integration_steps):
            self._time_cond.append(self._openpi0.model.time_embedding(t))
            t += self._delta_t

        # prior dist for sampling a_0 ~ N(0, I)
        self._latent_noise_dist = DiagMultivariateNormal(
            loc=torch.zeros(self._action_chunk_size * 7),
            scale=torch.ones(self._action_chunk_size * 7))

    def state_dict(self, *args, **kwargs):
        # Keep all non-frozen parameters.
        full_sd = super().state_dict(*args, **kwargs)

        # Keywords to keep per training mode
        mode_to_keys = {
            "full_action_head": {"action"},
            "lora_action_head": {"action_encoder", "action_decoder", "lora_"},
            "lora_lm_action_head": {
                "action_encoder", "action_decoder", "proprio_encoder", "lora_"
            },
            "lora_full": {
                "action_encoder", "action_decoder", "proprio_encoder", "lora_"
            },
            "lora_full_action_head": {"action", "proprio_encoder", "lora_"}
        }

        common_keys = {"_value_head", "_noise_head", "_loss.", "_optimizers."}

        keys_to_keep = mode_to_keys[self._train_mode] | common_keys

        return {
            k: v
            for k, v in full_sd.items()
            if any(key in k for key in keys_to_keep)
        }

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Normalize saved names from compiled runs and older wrapped checkpoints.
        state_dict = {
            key.removeprefix("_rl_alg.").replace("_orig_mod.", ""): value
            for key, value in state_dict.items()
        }
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def _denoising_mean(
            self, action: torch.Tensor, time_cond: torch.Tensor,
            action_inputs: Dict[str, torch.Tensor],
            caches: Dict[str, KVCache]) -> torch.Tensor:
        velocity = self._openpi0.compute_action_vel(
            action, time_cond, action_inputs, caches)
        return (action + velocity * self._delta_t).to(action.dtype)

    def _denoising_distribution(
            self, action: torch.Tensor, time_cond: torch.Tensor,
            action_inputs: Dict[str, torch.Tensor], caches: Dict[str, KVCache],
            hidden_states: Dict[str, torch.Tensor]) -> DiagMultivariateNormal:
        mean = self._denoising_mean(action, time_cond, action_inputs, caches)
        std = self._noise_head(hidden_states, action.float(), time_cond.float())
        return DiagMultivariateNormal(loc=mean.flatten(1), scale=std)

    @torch.inference_mode
    def _inference_step(self, time_step: TimeStep,
                        mode: Literal["predict", "rollout"]) -> AlgStep:
        """
        Common inference step.

        Args:
            time_step: ALF time step
            mode: Rollout samples each denoising step and records its log
                probability. Prediction follows the denoising means from the
                sampled initial noise.
        """
        bs = time_step.observation["rgb"].shape[0]
        pi0_obs = self._openpi0.create_openpi0_obs(time_step.observation)

        vlm_proprio_inputs, action_inputs = self._openpi0.prep_openpi0_inputs(
            pi0_obs, split_model_inputs=True)

        vision_embeds = ()
        if self._train_mode == "lora_lm_action_head":
            vision_embeds = self._openpi0.model._forward_siglip_and_text_embedding(
                vlm_proprio_inputs.pop("input_ids"),
                vlm_proprio_inputs.pop("pixel_values"),
            )
            hidden_states, vlm_proprio_kv_caches = (
                self._openpi0.model.
                get_vlm_proprio_hidden_state_with_vision_cache(
                    # Must clone the vision embeddings here as they are mutated in-place.
                    # Since we store them in a replay buffer.
                    vision_embeds.clone(),
                    **vlm_proprio_inputs,
                    return_caches=True))
        else:
            hidden_states, vlm_proprio_kv_caches = (
                self._openpi0.model.get_vlm_proprio_hidden_state(
                    **vlm_proprio_inputs, return_caches=True))

        convert_to_f32 = lambda x: x.to(torch.float32
                                        ) if x.dtype == torch.bfloat16 else x

        # Hidden states are always fed to f32 modules so we upcast here.
        hidden_states = alf.nest.map_structure(convert_to_f32, hidden_states)
        if self._action_head_dtype == torch.float32:
            # Upcast KVCaches to float32 only if action head is float32.
            for k in vlm_proprio_kv_caches.keys():
                vlm_proprio_kv_caches[k].key_cache = alf.nest.map_structure(
                    convert_to_f32, vlm_proprio_kv_caches[k].key_cache)
                vlm_proprio_kv_caches[k].value_cache = alf.nest.map_structure(
                    convert_to_f32, vlm_proprio_kv_caches[k].value_cache)

        action = self._latent_noise_dist.sample((bs,)).to(
            self._action_head_dtype).reshape(bs, self._action_chunk_size, 7)
        is_rollout = mode == "rollout"
        if is_rollout:
            value = self._value_head(hidden_states).squeeze(-1)
            log_prob = self._latent_noise_dist.log_prob(action.flatten(1))
            intermediate_actions = action.new_empty(
                (bs, self._num_integration_steps + 1, *action.shape[1:]))
            intermediate_actions[:, 0] = action

        for i, time_embedding in enumerate(self._time_cond):
            time_cond = time_embedding.expand(bs, -1)
            if is_rollout:
                dist = self._denoising_distribution(
                    action, time_cond, action_inputs, vlm_proprio_kv_caches,
                    hidden_states)
                action = dist.sample().reshape_as(action).to(self._action_head_dtype)
                # Score the same sample that replay will read.
                log_prob += dist.log_prob(action.flatten(1))
                intermediate_actions[:, i + 1] = action
            else:
                action = self._denoising_mean(
                    action, time_cond, action_inputs, vlm_proprio_kv_caches)

        new_info = ()
        if is_rollout:
            log_prob *= self._scale_log_prob

            # If just training the action head, we can cache the Paligemma forward
            if self._train_mode in ["full_action_head", "lora_action_head"]:
                # The KVCaches are classes but need to be represented as nested tensors
                # so that we can compute rollout info spec.
                # We'll convert it back later during train_step.
                kv_tensor_caches: Dict[str, List[torch.Tensor]] = {
                    k: [v.key_cache, v.value_cache]
                    for k, v in vlm_proprio_kv_caches.items()
                }
                vlm_proprio_inputs = dict()
            else:
                # If we're training the full model, we'll have to recompute everything.
                hidden_states = dict()
                kv_tensor_caches = dict()

            ppo_info = PPOInfo(log_prob=log_prob,
                               value=value,
                               step_type=time_step.step_type,
                               reward=time_step.reward,
                               discount=time_step.discount)

            new_info = RenOpenPi0PPOFlowInfo(
                hidden_states=hidden_states,
                vlm_proprio_inputs=vlm_proprio_inputs,
                vlm_proprio_kv_caches=kv_tensor_caches,
                vision_embeds=vision_embeds,
                action_inputs=action_inputs,
                intermediate_actions=intermediate_actions,
                ppo=ppo_info)

            # Need to convert to CPU (especially if caching hidden_states + kv caches).
            # Otherwise, GPU memory will steadily rise.
            # If caching large tensors into the buffer, we also want to avoid unnecessary
            # memory spikes during whole replay buffer training.
            # This can be done by setting ReplayBuffer.gather_all.convert_to_default_device to False.
            new_info = alf.nest.map_structure(
                lambda x: x.cpu()
                if isinstance(x, torch.Tensor) else x, new_info)

        if self._openpi0.model.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self._openpi0.model.final_action_clip_value,
                self._openpi0.model.final_action_clip_value,
            )

        action = self._openpi0.postprocess(action)

        return AlgStep(output=action, state=(), info=new_info)

    def predict_step(self, time_step: TimeStep, _) -> AlgStep:
        return self._inference_step(time_step, mode="predict")

    def rollout_step(self, time_step: TimeStep, _) -> AlgStep:
        return self._inference_step(time_step, mode="rollout")

    def preprocess_experience(self, time_step: TimeStep,
                              rollout_info: RenOpenPi0PPOFlowInfo, batch_info):
        time_step, ppo_info = super().preprocess_experience(
            time_step, rollout_info.ppo, batch_info)
        rollout_info = rollout_info._replace(ppo=ppo_info)
        return time_step, rollout_info

    def train_step(self, time_step: TimeStep, _,
                   rollout_info: RenOpenPi0PPOFlowInfo):
        ppo_info = rollout_info.ppo
        hidden_states = rollout_info.hidden_states
        vision_embeds = rollout_info.vision_embeds
        kv_tensor_caches = rollout_info.vlm_proprio_kv_caches
        action_inputs = rollout_info.action_inputs
        vlm_proprio_inputs = rollout_info.vlm_proprio_inputs
        intermediate_actions = rollout_info.intermediate_actions.to(
            self._action_head_dtype)

        if self._train_mode in ["full_action_head", "lora_action_head"]:
            # Recreate and reuse the KVCaches from the nested tensors from rollout.
            vlm_proprio_kv_caches: Dict[str, KVCache] = {
                k: KVCache(v[0], v[1])
                for k, v in kv_tensor_caches.items()
            }
        elif self._train_mode == "lora_lm_action_head":
            # Recompute PaliGemma forward minus vision tower
            hidden_states, vlm_proprio_kv_caches = (
                self._openpi0.model.
                get_vlm_proprio_hidden_state_with_vision_cache(
                    # Must clone the vision embeddings here as they are mutated in-place.
                    # Since we store them in a replay buffer.
                    vision_embeds.clone(),
                    **vlm_proprio_inputs,
                    return_caches=True))
        else:
            # Recompute PaliGemma forward.
            hidden_states, vlm_proprio_kv_caches = (
                self._openpi0.model.get_vlm_proprio_hidden_state(
                    **vlm_proprio_inputs, return_caches=True))

        hidden_states = alf.nest.map_structure(lambda x: x.detach().float(),
                                               hidden_states)

        value = self._value_head(hidden_states).squeeze(-1)
        bs = value.shape[0]
        log_prob = self._latent_noise_dist.log_prob(
            intermediate_actions[:, 0].flatten(1))
        entropy = ()
        if self._entropy_regularization is not None:
            entropy = self._latent_noise_dist.entropy().repeat(bs)
        for i, time_embedding in enumerate(self._time_cond):
            dist = self._denoising_distribution(
                intermediate_actions[:, i], time_embedding.expand(bs, -1),
                action_inputs, vlm_proprio_kv_caches, hidden_states)
            log_prob += dist.log_prob(intermediate_actions[:, i + 1].flatten(1))

            if self._entropy_regularization is not None:
                entropy += dist_utils.entropy_with_fallback(
                    dist, return_sum=False)[1]

        log_prob *= self._scale_log_prob

        # Entropy is originally scaled by (1 / (1 + K)) in the paper.
        if self._entropy_regularization is not None:
            entropy /= (1 + self._num_integration_steps)

        ppo_info = ppo_info._replace(log_prob=log_prob,
                                     entropy=entropy,
                                     value=value)

        return AlgStep(info=ppo_info, state=())

    # Override this as PPOAlgorithm implements after_update
    def after_update(self, time_step: TimeStep, info: RenOpenPi0PPOFlowInfo):
        pass

