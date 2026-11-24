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

from omegaconf import OmegaConf, DictConfig
import numpy as np

import os
from inspect import getfile
from pathlib import Path
from loguru import logger

from typing import Callable, Literal, Tuple, Dict, Union
import torch
from huggingface_hub import hf_hub_download

import alf
from alf.data_structures import TimeStep, AlgStep
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf import TensorSpec, NestedBoundedTensorSpec

from src.algorithms.bridge_adapter import ManiSkillBridgeAdapter
from src.environment.instructions import UTF8Encoder

from ren_openpi0.model.vla.pizero import PiZero
from ren_openpi0.agent.env_adapter.base import BaseEnvAdapter
from ren_openpi0.model.kv_cache import KVCache

RenOpenPi0Modules = Literal['vision', 'vlm', 'proprio', 'action']
RenOpenPi0TrainConfig = Literal["full_action_head", "lora_action_head",
                                "lora_vlm", "lora_full",
                                "lora_full_action_head"]


def get_train_param_config(
    train_mode: RenOpenPi0TrainConfig
) -> Tuple[Tuple[RenOpenPi0Modules, ...], Tuple[RenOpenPi0Modules, ...]]:
    """
    Helper function for getting common training parameter configs for fine-tuning RenOpenPi0Algorithm.

    Returns:
        A tuple of tuples containing modules to be frozen and LoRAed, respectively.
    """
    if train_mode in ["full_action_head", "lora_action_head"]:
        freeze_modules: Tuple[RenOpenPi0Modules,
                              ...] = ("vision", "vlm", "proprio")
        # This will be ignored if lora_rank is None for full_action_head
        lora_modules: Tuple[RenOpenPi0Modules, ...] = ("action", )
    elif train_mode == "lora_lm_action_head":
        freeze_modules = ("vision", )
        lora_modules = ("vlm", "proprio", "action")
    elif train_mode == "lora_full":
        freeze_modules = ()
        lora_modules = ("vision", "vlm", "proprio", "action")
    elif train_mode == "lora_full_action_head":
        freeze_modules = ()
        lora_modules = ("vision", "vlm", "proprio")
    else:
        raise ValueError(f"Invalid train_mode: {train_mode}")
    return freeze_modules, lora_modules


@alf.configurable
class RenOpenPi0Algorithm(RLAlgorithm):
    """
    Alf RLAlgorithm wrapper over RenOpenPi0 from the following repo:
    https://github.com/allenzren/open-pi-zero
    """

    def __init__(self,
                 observation_spec,
                 action_spec: NestedBoundedTensorSpec,
                 env_adapter_ctor: Callable[..., ManiSkillBridgeAdapter],
                 reward_spec: TensorSpec = TensorSpec(()),
                 default_language_instruction: str = None,
                 action_chunk_size: int = 4,
                 num_integration_steps: int = 10,
                 encoder_dtype: torch.dtype = torch.bfloat16,
                 action_head_dtype: torch.dtype = torch.bfloat16,
                 mode: Literal["train", "eval"] = "train",
                 torch_compile: bool = False,
                 lora_rank: int = None,
                 lora_target_modules: Tuple[RenOpenPi0Modules,
                                            ...] = ("vision", "vlm", "proprio",
                                                    "action"),
                 freeze_modules: Tuple[RenOpenPi0Modules, ...] = (),
                 use_grad_checkpointing: bool = False,
                 env=None,
                 config=None,
                 debug_summaries: bool = False,
                 name: str = "RenOpenPi0Algorithm"):
        """
        Initializes the RenOpenPi0Algorithm.

        Args:
            observation_spec: The observation spec of the environment.
            action_spec: The action spec of the environment.
            env_adapter_ctor: The constructor for the environment adapter.
            reward_spec: The reward spec of the environment.
            default_language_instruction: The default language instruction to be given
                to the model. If one is provided, will override all other language sources.
            action_chunk_size: The number of actions to be predicted in each chunk.
            num_integration_steps: The number of integration steps.
            encoder_dtype: The dtype to use for Paligemma + proprio encoder.
            action_head_dtype: The dtype to use for the action head.
            mode: The mode to run the algorithm in.
                - If "train": Weights will not be frozen. If LoRA is enabled,
                   all non-LoRA weights will be frozen. Model will be put in train mode.
                - If "eval": All weights will be frozen and model will be put in eval mode.
            torch_compile: Whether to torch compile the model. Results in a ~2x inference
                boost for batch size 1.
            lora_rank: The rank of the LoRA weights. If None, no LoRA is used.
            lora_target_modules: The modules to apply LoRA to. Users can selectively choose
                modules based off of the RenOpenPi0Modules Literal. When a module is specified for
                LoRA, all non-LoRA weights in that module will be frozen (except for
                certain encoder / decoder layers). Other unspecified modules will not be frozen
                by default and can be via freeze_modules. Note that lora_target_modules and
                freeze_modules can have no overlaps.
            freeze_modules: List of modules to freeze.
            use_grad_checkpointing: Whether to enable grad checkpointing.
            env: unused
            config: unused
            debug_summaries: unused
            name: unused
        """
        if lora_rank is not None and freeze_modules != ():
            assert not set(lora_target_modules) & set(freeze_modules), \
                "Modules defined as lora modules should not be specified by freeze_modules."

        lora_target_modules = () if lora_rank is None else lora_target_modules

        # Need to set this env variable so that paligemma weights / tokenizer can be found.
        # Cluster has a different explicit TRANSFORMERS_CACHE defined.
        hub_directory = Path.home() / ".cache/huggingface/hub"
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_directory))

        cfg, env_adapter = self._initialize_config_and_env_adapter(
            env_adapter_ctor)
        self._setup_params(cfg, action_chunk_size, num_integration_steps)
        self._define_lora_weights(cfg, lora_rank, lora_target_modules)

        # We'll load the model into cpu first.
        model = PiZero(cfg, use_grad_checkpointing=use_grad_checkpointing).to(
            encoder_dtype).to("cpu")
        torch.cuda.empty_cache()

        model.action_encoder.to(action_head_dtype)
        model.action_decoder.to(action_head_dtype)
        model.joint_model.mixtures["action"].to(action_head_dtype)

        # Load the pretrained model weights and then move to desired device.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for running ren_openpi0.")
        self._load_model_weights(model)
        # For now, we always default to cuda:0. Support specifying the device in the future if needed.
        model.to("cuda:0")

        if mode == "eval":
            model.freeze_all_weights()
            model.eval()
        elif mode == "train":
            # The training code of RenOpenPi0 actually uses the same
            # weights for the action and proprio head. We'll separate it
            # to make targeted module training transparent.
            # model.tie_action_proprio_weights()

            # This freezes unused weights from the VLM and the token embedding layer.
            model.freeze_unused_weights()

            self.freeze_weights_for_training(model,
                                             lora_modules=lora_target_modules,
                                             freeze_modules=freeze_modules)

            model.train()

            trainable = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(
                f"RenOpenPi0 trainable params ratio: {trainable} / {total}: {trainable/total:.4f}"
            )

        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         reward_spec=reward_spec,
                         train_state_spec=(),
                         env=env,
                         config=config,
                         debug_summaries=debug_summaries,
                         is_on_policy=False,
                         name=name)

        self._model = model
        self._torch_compile = torch_compile
        self._compiled = False
        self._encoder_dtype = encoder_dtype
        self._action_head_dtype = action_head_dtype
        self._env_adapter = env_adapter
        self._default_instruction = default_language_instruction


    @property
    def model(self) -> PiZero:
        return self._model

    def _compile_model(self):
        # For some reason, torch compiling the overall model doesn't trigger the
        # compile, so we'll do it individually on the vision tower and joint model.
        torch.set_float32_matmul_precision('high')
        self._model.vision_tower = torch.compile(self._model.vision_tower,
                                                 mode='default')
        self._model.joint_model = torch.compile(self._model.joint_model,
                                                mode='default')

    @staticmethod
    def _initialize_config_and_env_adapter(
        env_adapter_ctor: Callable[..., ManiSkillBridgeAdapter]
    ) -> Tuple[DictConfig, BaseEnvAdapter]:
        """
        Loads configs and instantiates the environment adapter.

        Returns:
            A tuple of
                cfg: The config for the RenOpenPi0 model.
                env_adapter: The environment adapter.
        """
        # The model module sits at ren_openpi0/model/vla/pizero.py in its checkout.
        config_dir = Path(getfile(PiZero)).resolve().parents[3] / "config"
        train_cfg = OmegaConf.load(config_dir / "train/bridge.yaml")

        # The eval config holds the env adapter configs which we need for RL.
        eval_cfg = OmegaConf.load(config_dir / "eval/bridge.yaml")

        adapter_cfg = eval_cfg.env.adapter

        dataset_statistics_path = Path(adapter_cfg.dataset_statistics_path)
        if not dataset_statistics_path.is_absolute():
            dataset_statistics_path = config_dir / dataset_statistics_path.name

        pretrained_model_path = str(adapter_cfg.pretrained_model_path)
        if "${oc.env:" in pretrained_model_path:
            # Simple `${oc.env:VAR}/suffix` expansion to avoid depending on
            # Hydra instantiation while still supporting the existing config.
            start = pretrained_model_path.find("${oc.env:") + len("${oc.env:")
            end = pretrained_model_path.find("}", start)
            var_name = pretrained_model_path[start:end]
            prefix = os.environ.get(var_name, "")
            suffix = pretrained_model_path[end + 1:]
            if prefix:
                pretrained_model_path = prefix + suffix

        if not os.path.isdir(pretrained_model_path):
            # Fall back to the public HuggingFace repo if the local cache
            # path is not configured or does not contain the model.
            model_name = Path(pretrained_model_path).name
            pretrained_model_path = f"google/{model_name}"

        env_adapter = env_adapter_ctor(
            dataset_statistics_path=str(dataset_statistics_path),
            pretrained_model_path=pretrained_model_path,
            tokenizer_padding=adapter_cfg.tokenizer_padding,
            num_image_tokens=adapter_cfg.num_image_tokens,
            image_size=tuple(adapter_cfg.image_size),
            max_seq_len=adapter_cfg.max_seq_len)

        env_adapter.reset()

        return train_cfg, env_adapter

    @staticmethod
    def _setup_params(cfg: DictConfig, action_chunk_size: int,
                      num_integration_steps: int):
        """
        Function for overwriting certain parameters.
        If more parameters from the hydra config are of interest, they can be exposed
        via this function.
        """
        cfg.horizon_steps = action_chunk_size
        cfg.num_integration_steps = num_integration_steps

        # The pretrained ckpt was trained with these settings so these must be set to the same values
        cfg.time_max_period = 10000.0
        cfg.action_expert_rope_theta = 10000.0

    @staticmethod
    def _define_lora_weights(cfg: DictConfig, lora_rank: int,
                             lora_target_modules: Tuple[RenOpenPi0Modules,
                                                        ...]):
        """
        Helper function for defining LoRA weights.
        Will modify the provided config in-place.

        Args:
            cfg: The config for the RenOpenPi0 model.
            lora_rank: The rank of the LoRA weights. If None, no LoRA is used.
            lora_target_modules: The modules to apply LoRA to.
        """
        if lora_rank is not None:
            cfg.vision.config.lora.r = lora_rank
            cfg.vision_projector.config.lora.r = lora_rank
            cfg.joint.config.lora.r = lora_rank

            cfg.vision.use_lora = "vision" in lora_target_modules
            cfg.vision_projector.use_lora = "vision" in lora_target_modules
            cfg.joint.config.mixture.vlm.use_lora = "vlm" in lora_target_modules
            cfg.joint.config.mixture.proprio.use_lora = "proprio" in lora_target_modules
            cfg.joint.config.mixture.action.use_lora = "action" in lora_target_modules

    @staticmethod
    def _freeze_module(module: torch.nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def _freeze_all_non_lora_module(module: torch.nn.Module):
        for name, param in module.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    def freeze_weights_for_training(self, model: PiZero,
                                    lora_modules: Tuple[RenOpenPi0Modules,
                                                        ...],
                                    freeze_modules: Tuple[RenOpenPi0Modules,
                                                          ...]):
        if "vision" in lora_modules:
            self._freeze_all_non_lora_module(model.vision_tower)
            self._freeze_all_non_lora_module(model.multi_modal_projector)
        if "vlm" in lora_modules:
            self._freeze_all_non_lora_module(model.joint_model.mixtures["vlm"])
        if "proprio" in lora_modules:
            # Note that this does not freeze the proprio_encoder!
            self._freeze_all_non_lora_module(
                model.joint_model.mixtures["proprio"])
        if "action" in lora_modules:
            # Note that this does not freeze the action_encoder and action_decoder!
            self._freeze_all_non_lora_module(
                model.joint_model.mixtures["action"])

        if "vision" in freeze_modules:
            self._freeze_module(model.vision_tower)
            self._freeze_module(model.multi_modal_projector)
        if "vlm" in freeze_modules:
            self._freeze_module(model.joint_model.mixtures["vlm"])
        if "proprio" in freeze_modules:
            self._freeze_module(model.proprio_encoder)
            self._freeze_module(model.joint_model.mixtures["proprio"])
        if "action" in freeze_modules:
            self._freeze_module(model.action_encoder)
            self._freeze_module(model.action_decoder)
            self._freeze_module(model.joint_model.mixtures["action"])

    @staticmethod
    def _load_model_weights(model):
        """
        Helper function for loading model weights.

        Arg:
            model: The model to load weights into.
        """
        ckpt_path = None
        cache_dir = os.environ.get("TRANSFORMERS_CACHE")
        if cache_dir:
            candidate = Path(cache_dir) / "bridge_beta.pt"
            if candidate.is_file():
                ckpt_path = candidate

        if ckpt_path is None:
            ckpt_path = Path(
                hf_hub_download(
                    repo_id="allenzren/open-pi-zero",
                    filename="bridge_beta_step19296_2024-12-26_22-30_42.pt",
                    cache_dir=cache_dir))

        data = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        data["model"] = {
            k.replace("_orig_mod.", ""): v
            for k, v in data["model"].items()
        }
        model.load_state_dict(data["model"], strict=False)

    def create_openpi0_obs(
            self, raw_obs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        if self._default_instruction is not None:
            instructions = [self._default_instruction] * raw_obs["rgb"].shape[0]
        else:
            encoded = raw_obs.get("operator_instruction", raw_obs.get("instrs"))
            if encoded is None:
                raise ValueError("The observation is missing a language instruction.")
            instructions = UTF8Encoder.decode(encoded)

        # PiZero needs the RGB image and joint positions
        pi0_obs = {
            # Convert to channel last
            "rgb": raw_obs["rgb"].permute(0, 2, 3, 1).cpu().numpy(),
            "joint_pos": raw_obs["joint_pos"].cpu().numpy(),
            "instrs": instructions,
        }
        return pi0_obs

    def prep_openpi0_inputs(
        self,
        pi0_obs: Dict[str, np.ndarray],
        split_model_inputs: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], ...]]:
        """
        Helper function for preparing inputs for RenOpenPi0 action inference.

        Args:
            pi0_obs: The observation from create_openpi0_obs
            split_model_inputs: If True, will provide two separate dictionaries,
                one for VLM + proprio forward and the other for action forward.

            Returns:
                Dictionary of model input tensors. Can be two separate dictionaries
                depending on split_model_inputs.
        """
        # We lazy torch compile the model because alf imports weights after the __init__
        # and weights will not be properly imported on a torch compiled model.
        if self._torch_compile and not self._compiled:
            self._compile_model()
            self._compiled = True

        inputs: Dict[str, torch.Tensor] = self._env_adapter.preprocess(pi0_obs)

        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self._model.build_causal_mask_and_position_ids(
                inputs["attention_mask"], dtype=self._encoder_dtype))

        image_text_proprio_mask, action_mask = (
            self._model.split_full_mask_into_submasks(causal_mask))

        vlm_proprio_inputs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"].to(self._encoder_dtype),
            "image_text_proprio_mask": image_text_proprio_mask,
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "proprios": inputs["proprios"].to(self._encoder_dtype),
        }

        action_inputs = {
            "action_mask": action_mask,
            "action_position_ids": action_position_ids,
        }

        if split_model_inputs:
            return vlm_proprio_inputs, action_inputs

        return vlm_proprio_inputs | action_inputs

    def forward_actions(self,
                        model_inputs: Dict[str, torch.Tensor],
                        vlm_proprio_kv_caches: Dict[str, 'KVCache'] = None,
                        action0: torch.Tensor = None) -> torch.Tensor:
        """
        Helper function for computing robot actions.

        Args:
            model_inputs: Model inputs from prep_openpi0_inputs
            vlm_proprio_kv_caches: Optional KV caches in case cached values
                are available. If not provided, they will be computed.
            action0: Optional starting noise vector. If not provided,
                it will be sampled from ~ N(0, I).
        """
        if vlm_proprio_kv_caches is None:
            actions = self._model.infer_action(**model_inputs, action0=action0)
        else:
            actions = self._model.infer_action_with_kv(
                **model_inputs,
                vlm_proprio_kv_caches=vlm_proprio_kv_caches,
                action0=action0)

        return self.postprocess(actions)

    def postprocess(self, actions: torch.Tensor):
        # TODO: make env_adapter a torch-based
        return torch.tensor(
            self._env_adapter.postprocess(actions.float().cpu().numpy()))

    def _step(self, time_step: TimeStep) -> torch.Tensor:
        pi0_obs = self.create_openpi0_obs(time_step.observation)
        model_inputs = self.prep_openpi0_inputs(pi0_obs)
        env_actions = self.forward_actions(model_inputs)
        return env_actions

    def compute_action_vel(
            self, action: torch.Tensor, time_cond: torch.Tensor,
            action_inputs: Dict[str, torch.Tensor],
            vlm_proprio_kv_caches: Dict[str, KVCache]) -> torch.Tensor:
        r"""
        Helper function to compute the instantaneous flow of the action.

        Args:
            action: The current (B, C, 7) action at time step t \in [0, 1].
            time_cond: The (B, T) time embedding at time step t \in [0, 1].
            action_inputs: The action inputs.
            vlm_proprio_kv_caches: The vlm and proprio KV caches.

        Returns:
            The (B, C, 7) instantaneous flow of the action at time step t \in [0, 1].
        """
        # [B, C, 7]
        action_embeds = self.model.action_encoder(action, time_cond)

        # [B, C, 7]
        action_embeds = self.model.joint_model(
            attention_mask=action_inputs["action_mask"],
            position_ids_all={"action": action_inputs["action_position_ids"]},
            embeds_all={"action": action_embeds},
            time_cond=time_cond,
            kv_caches=vlm_proprio_kv_caches,
            cache_mode=
            "append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
        )["action"]
        # [B, C, 7]
        action_vel = self.model.action_decoder(action_embeds)
        return action_vel

    def predict_step(self, time_step: TimeStep, state) -> AlgStep:
        return AlgStep(output=self._step(time_step), state=state)
