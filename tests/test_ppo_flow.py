# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import torch
from alf.data_structures import TimeStep
from alf.utils.dist_utils import DiagMultivariateNormal

from src.algorithms.ppo_flow import RenOpenPi0PPOFlowAlgorithm


class PPOFlowTest(unittest.TestCase):

    def test_replay_matches_rollout_log_probability(self):
        batch_size, chunk_size, action_size = 3, 2, 7
        flat_action_size = chunk_size * action_size
        time_step = TimeStep(
            observation={"rgb": torch.zeros(batch_size, 3, 2, 2)},
            step_type=torch.zeros(batch_size, dtype=torch.int32),
            reward=torch.zeros(batch_size),
            discount=torch.ones(batch_size))

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.random.fork_rng():
                torch.manual_seed(42)
                # Use fixed model outputs to exercise rollout and replay on CPU.
                algorithm = RenOpenPi0PPOFlowAlgorithm.__new__(
                    RenOpenPi0PPOFlowAlgorithm)
                torch.nn.Module.__init__(algorithm)
                algorithm._openpi0 = SimpleNamespace(
                    create_openpi0_obs=lambda obs: obs,
                    prep_openpi0_inputs=lambda obs, **kwargs: ({}, {}),
                    model=SimpleNamespace(
                        get_vlm_proprio_hidden_state=lambda **kwargs: ({}, {}),
                        final_action_clip_value=None),
                    compute_action_vel=lambda action, *args: 0.25 * action + 0.05,
                    postprocess=lambda action: action)
                algorithm._value_head = lambda _: torch.zeros(batch_size, 1)
                algorithm._noise_head = lambda *_: torch.full(
                    (batch_size, flat_action_size), 0.09)
                algorithm._train_mode = "lora_full_action_head"
                algorithm._action_head_dtype = dtype
                algorithm._action_chunk_size = chunk_size
                algorithm._num_integration_steps = 2
                algorithm._time_cond = [torch.zeros(1, 2), torch.ones(1, 2)]
                algorithm._delta_t = 0.5
                algorithm._scale_log_prob = 0.2
                algorithm._entropy_regularization = None
                algorithm._latent_noise_dist = DiagMultivariateNormal(
                    loc=torch.zeros(flat_action_size),
                    scale=torch.ones(flat_action_size))

                rollout = algorithm.rollout_step(time_step, ())
                replay = algorithm.train_step(time_step, (), rollout.info)

                torch.testing.assert_close(
                    replay.info.log_prob, rollout.info.ppo.log_prob)
