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
from jumanji.environments import CVRP, TSP, BinPack, Knapsack, MultiCVRP
# from jumanji.environments.logic.sliding_tile_puzzle.reward import SparseRewardFn
from jumanji.wrappers import MultiToSingleWrapper, VmapAutoResetWrapper
# from jumanji.environments.logic.sliding_tile_puzzle.reward import (
#   SparseRewardFn as STP_SparseRewardFn,
# )

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
from jumanji.training.networks.base import FeedForwardNetwork
import jax
import jax.numpy as jnp
from jumanji.training.networks.base import FeedForwardNetwork


def _copy_tree(tree):
    return jax.tree_util.tree_map(lambda x: x, tree)


def polyak_update(target, online, tau: float):
    # target <- (1-tau)*target + tau*online
    return jax.tree_util.tree_map(lambda t, s: (1.0 - tau) * t + tau * s, target, online)


class VAndDoubleQNetworks(FeedForwardNetwork):
    """
    1个V + (Q1,Q2) + (targetQ1,targetQ2)
    - 复用 policy_structure 作为 q_structure：输出 [B, A]
    - 通过 action index gather 成 Q(s,a): [B]
    - init(key, obs) 不变：兼容 jumanji 的 setup_training_state
    - 提供 apply(params, obs) 返回 V：兼容你现有 value_network.apply 调用
    """

    def __init__(self, v_structure: FeedForwardNetwork, q_structure: FeedForwardNetwork):
        self.v_structure = v_structure
        self.q_structure = q_structure

    def init(self, key, obs):
        key_v, key_q1, key_q2 = jax.random.split(key, 3)

        v_params = self.v_structure.init(key_v, obs)
        q1_params = self.q_structure.init(key_q1, obs)
        q2_params = self.q_structure.init(key_q2, obs)

        # ✅ targetQ 必须从 onlineQ 拷贝初始化（一开始相同最稳）
        tq1_params = _copy_tree(q1_params)
        tq2_params = _copy_tree(q2_params)

        return {"v": v_params, "q1": q1_params, "q2": q2_params, "tq1": tq1_params, "tq2": tq2_params}

    # ✅ 兼容：旧代码 value_network.apply(params, obs) 继续返回 V
    def apply(self, params, obs):
        return self.v_structure.apply(params["v"], obs)

    def apply_v(self, params, obs):
        return self.v_structure.apply(params["v"], obs)

    def _gather_q(self, q_all, act):
        # q_all: [B, A]   act: [B] (int)
        act = act.astype(jnp.int32)
        return jnp.take_along_axis(q_all, act[..., None], axis=-1)[..., 0]  # [B]

    def apply_q(self, params, obs, act):
        # q_structure 输出 [B, A]，我们 gather 得到 [B]
        q1_all = self.q_structure.apply(params["q1"], obs)
        q2_all = self.q_structure.apply(params["q2"], obs)
        tq1_all = self.q_structure.apply(params["tq1"], obs)
        tq2_all = self.q_structure.apply(params["tq2"], obs)

        q1 = self._gather_q(q1_all, act)
        q2 = self._gather_q(q2_all, act)
        tq1 = self._gather_q(tq1_all, act)
        tq2 = self._gather_q(tq2_all, act)
        return q1, q2, tq1, tq2

    # targetQ 的慢更新（你在 dhvl 的每个 update step 之后调用）
    def update_target_q(self, params, tau: float):
        params = dict(params)
        params["tq1"] = polyak_update(params["tq1"], params["q1"], tau)
        params["tq2"] = polyak_update(params["tq2"], params["q2"], tau)
        return params


class DoubleValueNetwork:
    def __init__(self, base):
        self.base = base

    def init(self, key, obs):
        params_v = self.base.init(key, obs)
        return {
            "value": params_v,
            "delayed_value": jax.tree_util.tree_map(lambda x: x, params_v),  #
        }

    def apply(self, params, obs):
        v1 = self.base.apply(params["value"], obs)
        v2 = self.base.apply(params["delayed_value"], obs)
        return v1, v2


def setup_env(cfg: DictConfig) -> Environment:
    if cfg.grpo.reward_mode == "sparse":
        # if cfg.env.name == "sliding_tile_puzzle":
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
    # policy_structure = actor_critic_networks.policy_network
    # value_structure = actor_critic_networks.value_network
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
        ppo_cfg = PPOConfig(
            clip_eps=cfg.ppo.clip_eps,
            normalize_adv=cfg.ppo.normalize_adv,
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
            ppo_cfg=ppo_cfg,
        )
    elif cfg.agent == "dhvl":
        # complex_critic = VAndDoubleQNetworks(
        #     v_structure=value_structure,
        #     q_structure=policy_structure  # 复用图纸，不共享参数
        # )
        # new_actor_critic_networks = actor_critic_networks._replace(
        #     policy_network=policy_structure,  # <--- Actor 用的 (第1个)
        #     value_network=complex_critic  # <--- Critic 用的 (第2-6个)
        # )
        from agent.dhvl import dhvlAgent, dhvlConfig
        optimizer = optax.adam(cfg.env.a2c.learning_rate)
        dhvl_cfg = dhvlConfig(
            clip_eps=cfg.dhvl.clip_eps,
            normalize_adv=cfg.ppo.normalize_adv,
            bootstrapping_factor=cfg.dhvl.bootstrapping_factor,
            update_epochs=cfg.dhvl.update_epochs,
            tau=cfg.dhvl.tau,
            minibatch_size=cfg.dhvl.minibatch_size,
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
            dhvl_cfg=dhvl_cfg,
        )
    elif cfg.agent == "dhvl_delay":
        # complex_critic = VAndDoubleQNetworks(
        #     v_structure=value_structure,
        #     q_structure=policy_structure  # 复用图纸，不共享参数
        # )
        # new_actor_critic_networks = actor_critic_networks._replace(
        #     policy_network=policy_structure,  # <--- Actor 用的 (第1个)
        #     value_network=complex_critic  # <--- Critic 用的 (第2-6个)
        # )
        from agent.dhvl_delayed import dhvlAgent, dhvlConfig
        ac = actor_critic_networks._replace(
            value_network=DoubleValueNetwork(actor_critic_networks.value_network))
        actor_critic_networks = ac


        optimizer = optax.adam(cfg.env.a2c.learning_rate)
        dhvl_cfg = dhvlConfig(
            clip_eps=cfg.dhvl.clip_eps,
            normalize_adv=cfg.ppo.normalize_adv,
            bootstrapping_factor=cfg.dhvl.bootstrapping_factor,
            value_delay=cfg.dhvl.value_delay,
            update_epochs=cfg.dhvl.update_epochs,
            tau=cfg.dhvl.tau,
            minibatch_size=cfg.dhvl.minibatch_size,
            tar_tau=cfg.dhvl.tar_tau,
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
            dhvl_cfg=dhvl_cfg,
        )



    else:
        raise ValueError(f"Expected agent name to be in ['random', 'a2c', 'grpo', 'PPO', 'dhvl'], got {cfg.agent}.")
