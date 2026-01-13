# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a2c copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import logging
from typing import Dict, Tuple

import hydra
import jax
import jax.numpy as jnp
import omegaconf
from tqdm.auto import trange

from jumanji.training import utils
from jumanji.training.agents.random import RandomAgent
from jumanji.training.loggers import TerminalLogger
from setup_train import setup_env, setup_agent, setup_evaluators, setup_training_state,setup_logger

from jumanji.training.timer import Timer
from jumanji.training.types import TrainingState


@hydra.main(config_path="../../configs/env", config_name="game_2048.yaml")
def train(cfg: omegaconf.DictConfig, log_compiles: bool = False) -> None:
    logging.info(omegaconf.OmegaConf.to_yaml(cfg))
    logging.getLogger().setLevel(logging.INFO)
    logging.info({"devices": jax.local_devices()})

    key, init_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
    logger = setup_logger(cfg)
    env = setup_env(cfg)
    agent = setup_agent(cfg, env)
    stochastic_eval, greedy_eval = setup_evaluators(cfg, agent)
    training_state = setup_training_state(env, agent, init_key)

    n_steps = int(cfg.env.training.n_steps)
    num_ls = int(cfg.env.training.num_learner_steps_per_epoch)
    B = int(cfg.env.training.total_batch_size)

    steps_per_env_per_epoch = n_steps * num_ls
    total_steps_per_epoch = steps_per_env_per_epoch * B
    num_steps_per_epoch = total_steps_per_epoch
    eval_timer = Timer(out_var_name="metrics")
    train_timer = Timer(out_var_name="metrics", num_steps_per_timing=num_steps_per_epoch)

    @functools.partial(jax.pmap, axis_name="devices")
    def epoch_fn(training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        training_state, metrics = jax.lax.scan(
            lambda training_state, _: agent.run_epoch(training_state),
            training_state,
            None,
            cfg.env.training.num_learner_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, metrics

    with jax.log_compiles(log_compiles), logger:
        for i in trange(
            cfg.env.training.num_epochs,
            disable=isinstance(logger, TerminalLogger),
        ):
            env_steps = i * num_steps_per_epoch

            # Evaluation
            key, stochastic_eval_key, greedy_eval_key = jax.random.split(key, 3)
            # Stochastic evaluation
            with eval_timer:
                metrics = stochastic_eval.run_evaluation(
                    training_state.params_state, stochastic_eval_key
                )
                jax.block_until_ready(metrics)
            logger.write(
                data=utils.first_from_device(metrics),
                label="eval_stochastic",
                env_steps=env_steps,
            )
            if not isinstance(agent, RandomAgent):
                # Greedy evaluation
                with eval_timer:
                    metrics = greedy_eval.run_evaluation(
                        training_state.params_state, greedy_eval_key
                    )
                    jax.block_until_ready(metrics)
                logger.write(
                    data=utils.first_from_device(metrics),
                    label="eval_greedy",
                    env_steps=env_steps,
                )

            # Training
            with train_timer:
                training_state, metrics = epoch_fn(training_state)
                jax.block_until_ready((training_state, metrics))

            train_log = utils.first_from_device(metrics)

            avg_steps_per_env = float(train_log.get("avg_steps_per_env", 0.0))
            avg_episodes_per_env = float(train_log.get("avg_episodes_per_env", 0.0))
            total_episodes = float(train_log.get("total_episodes", 0.0))
            train_log.update({
                "steps/avg_steps_per_env_actual": avg_steps_per_env,
                "steps/avg_episodes_per_env": avg_episodes_per_env,
                "steps/total_episodes": total_episodes,
            })
            logger.write(
                data=train_log,
                label="train",
                env_steps=env_steps,
            )


if __name__ == "__main__":
    train()