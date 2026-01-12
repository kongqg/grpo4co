# grpo4co/setup_train.py  —— 轻量包装器：复用官方 env/评测/状态，只重写 agent 选择
import jax
import jumanji
from omegaconf import DictConfig, OmegaConf
import optax

# 复用官方的工具与默认实现（不会去 import 任何具体环境，如 FlatPack）
from jumanji.training.setup_train import (
    setup_env as _setup_env,
    setup_logger as _setup_logger,
    _setup_actor_critic_neworks,
    _setup_random_policy,
)

from jumanji.env import Environment
from jumanji.training.agents.random import RandomAgent
from jumanji.training.agents.a2c import A2CAgent

from agent.grpo import GRPOAgent, GRPOConfig
from agent.ppo import PPOConfig, PPOAgent
from agent.dhvl import dhvlAgent, dhvlConfig
from jumanji.environments import CVRP,TSP, BinPack, Knapsack,MultiCVRP
#from jumanji.environments.logic.sliding_tile_puzzle.reward import SparseRewardFn
from jumanji.wrappers import MultiToSingleWrapper, VmapAutoResetWrapper
#from jumanji.environments.logic.sliding_tile_puzzle.reward import (
 #   SparseRewardFn as STP_SparseRewardFn,
#)

from jumanji.environments.routing.cvrp.reward import (
    SparseReward as CVRP_SparseReward,
)
from jumanji.environments.routing.tsp.reward import (
    SparseReward as TSP_SparseReward,
)
from jumanji.environments.packing.knapsack.reward import (
    SparseReward as KNAP_SparseReward,
)
from jumanji.environments.packing.bin_pack.reward import (
    SparseReward as BIN_SparseReward,
)
from jumanji.environments.routing.multi_cvrp.reward import (
    SparseReward as Multi_cvrp_SparseReward,
)

def setup_env(cfg: DictConfig) -> Environment:
    if cfg.grpo.reward_mode == "sparse":
        #if cfg.env.name == "sliding_tile_puzzle":
         #   jax.debug.print("sliding_tile_puzzle reward mode is sparse")
          #  env = SlidingTilePuzzle(reward_fn=STP_SparseRewardFn())
        if cfg.env.name == "cvrp":
            jax.debug.print("cvrp reward mode is sparse")
            env = CVRP(reward_fn=CVRP_SparseReward())
        elif cfg.env.name == "tsp":
            jax.debug.print("tsp reward mode is sparse")
            env = TSP(reward_fn=TSP_SparseReward())
        elif cfg.env.name == "bin_pack":
            jax.debug.print("bin_pack reward mode is sparse")
            env = BinPack(reward_fn=BIN_SparseReward())
        elif cfg.env.name == "knapsack":
            jax.debug.print("knapsack reward mode is sparse")
            env = Knapsack(reward_fn=KNAP_SparseReward())
        elif cfg.env.name == "multi_cvrp":
            jax.debug.print("multi_cvrp reward mode is sparse")
            env = MultiCVRP(reward_fn=Multi_cvrp_SparseReward(2, 20, 20))
        else:
            env = jumanji.make(cfg.env.registered_version)
    elif cfg.grpo.reward_mode == "dense":
        env = jumanji.make(cfg.env.registered_version)

    if cfg.env.name in {"lbf", "connector", "search_and_rescue"}:
        env = MultiToSingleWrapper(env)
    env = VmapAutoResetWrapper(env)
    return env

def setup_evaluators(cfg, agent):
    from jumanji.training.setup_train import setup_evaluators as _setup_evaluators
    return _setup_evaluators(cfg, agent)

def setup_training_state(env, agent, init_key):
    from jumanji.training.setup_train import setup_training_state as _setup_training_state
    return _setup_training_state(env, agent, init_key)

def setup_logger(cfg: DictConfig):
    return _setup_logger(cfg)


def setup_agent(cfg: DictConfig, env: Environment):
    cfg_net = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True)
    )
    cfg_net.agent = "a2c"
    actor_critic_networks = _setup_actor_critic_neworks(cfg_net, env)
    if cfg.agent == "random":
        random_policy = _setup_random_policy(cfg, env)
        return RandomAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            random_policy=random_policy,
        )

    elif cfg.agent == "a2c":
        actor_critic_networks = _setup_actor_critic_neworks(cfg, env)
        optimizer = optax.adam(cfg.env.a2c.learning_rate)
        return A2CAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=cfg.env.a2c.normalize_advantage,
            discount_factor=cfg.env.a2c.discount_factor,
            bootstrapping_factor=cfg.env.a2c.bootstrapping_factor,
            l_pg=cfg.env.a2c.l_pg,
            l_td=cfg.env.a2c.l_td,
            l_en=cfg.env.a2c.l_en,
        )

    elif cfg.agent == "grpo":
        optimizer = optax.adam(cfg.env.a2c.learning_rate)
        grpo_cfg = GRPOConfig(
            clip_eps=cfg.grpo.clip_eps,
            supervision_mode=cfg.grpo.supervision_mode,
            kl_coef=cfg.grpo.kl_coef,
            percentile_p=cfg.grpo.percentile_p,
            mean_reward_method=cfg.grpo.mean_reward_method,
            reward_mode=cfg.grpo.reward_mode
        )
        return GRPOAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=False,
            discount_factor=cfg.env.a2c.discount_factor,
            bootstrapping_factor=cfg.env.a2c.bootstrapping_factor,
            l_pg=cfg.env.a2c.l_pg,
            l_td=cfg.env.a2c.l_td,
            l_en=cfg.env.a2c.l_en,
            grpo_cfg=grpo_cfg,
        )
    elif cfg.agent == "ppo":
        optimizer = optax.adam(cfg.env.a2c.learning_rate)
        grpo_cfg = PPOConfig(
            clip_eps=cfg.ppo.clip_eps,
            normalize_adv=cfg.ppo.normalize_adv,
            kl_coef=cfg.ppo.kl_coef,
        )
        return PPOAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=False,
            discount_factor=cfg.env.a2c.discount_factor,
            bootstrapping_factor=cfg.env.a2c.bootstrapping_factor,
            l_pg=cfg.env.a2c.l_pg,
            l_td=cfg.env.a2c.l_td,
            l_en=cfg.env.a2c.l_en,
            grpo_cfg=grpo_cfg,
        )
    elif cfg.agent == "dhvl":
        optimizer = optax.adam(cfg.env.a2c.learning_rate)
        dhvl_cfg = dhvlConfig(
            clip_eps=cfg.ppo.clip_eps,
            normalize_adv=cfg.ppo.normalize_adv,
            kl_coef=cfg.ppo.kl_coef,
        )
        return dhvlAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=False,
            discount_factor=cfg.env.a2c.discount_factor,
            bootstrapping_factor=cfg.env.a2c.bootstrapping_factor,
            l_pg=cfg.env.a2c.l_pg,
            l_td=cfg.env.a2c.l_td,
            l_en=cfg.env.a2c.l_en,
            grpo_cfg=dhvl_cfg,
        )


    else:
        raise ValueError(f"Expected agent name to be in ['random', 'a2c', 'grpo', 'PPO', 'dhvl'], got {cfg.agent}.")
