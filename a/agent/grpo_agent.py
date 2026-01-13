# ppo_by_steps.py

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import rlax
from chex import dataclass
from jumanji.training.agents.a2c import A2CAgent
from jumanji.training.types import (
    ActingState
)
from jumanji.training.types import Transition
from jumanji.types import StepType


@dataclass
class GRPOConfig:
    # GRPO core
    clip_eps: float = 0.1

    # normalization
    normalize_adv: bool = True

    # GRPO 监督方式
    supervision_mode: str = "outcome"  # 默认为 ppo_by_steps，可以被配置文件覆盖
    num_policy_updates: int = 3
    kl_coef: float = 0.0


# ===================================================================
# GRPO Agent 实现
# ===================================================================

class GRPOAgent(A2CAgent):
    """
    一个继承自 Jumanji A2C Agent 的 GRPO Agent。
    它复用了 A2C 的大部分结构，但重写了损失函数来匹配 GRPO 的三种监督模式。
    """

    def __init__(self, grpo_cfg: GRPOConfig, **kwargs: Dict):
        """
        初始化 GRPO Agent。
        Args:
            grpo_cfg: 包含 GRPO 特定超参数的配置对象。
            **kwargs: 所有传递给父类 A2CAgent 的标准参数。
        """
        # 调用父类 (A2CAgent) 的构造函数，完成所有标准初始化
        super().__init__(**kwargs)

        # 保存您自己的 GRPO 特定配置
        self.grpo_cfg = grpo_cfg
        self.batch_size = int(getattr(self, "total_batch_size",
                                      kwargs.get("total_batch_size")))

    def _loss_from_data(
            self,
            params,  # ActorCriticParams（含 .actor / .critic）
            acting_state,  # ActingState（含 .key 等）
            data,  # Transition：observation, next_observation, action, reward, discount, log_prob(旧策略)
    ):
        import functools
        import jax
        import jax.numpy as jnp
        import rlax
        from jax import debug as jdebug

        # ========== 1) 价值网络（仅 ppo_by_steps 模式用；其余模式仅监控） ==========
        value_apply = self.actor_critic_networks.value_network.apply
        last_observation = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        observation_for_value = jax.tree_util.tree_map(
            lambda obs_0_tm1, obs_t: jnp.concatenate([obs_0_tm1, obs_t[None]], axis=0),
            data.observation,
            last_observation,
        )
        value = jax.vmap(value_apply, in_axes=(None, 0))(params.critic, observation_for_value)  # (T+1,B)
        discounts = jnp.asarray(self.discount_factor * data.discount, dtype=jnp.float32)  # (T,B)

        # ========== 2) advantage（按 supervision_mode）+ 调试打印 ==========
        if self.grpo_cfg.supervision_mode == "ppo_by_steps":
            advantage = jax.vmap(
                functools.partial(rlax.td_lambda, lambda_=self.bootstrapping_factor, stop_target_gradients=True),
                in_axes=1, out_axes=1,
            )(v_tm1=value[:-1], r_t=data.reward, discount_t=discounts, v_t=value[1:])
            td_targets = advantage + value[:-1]
            critic_loss = jnp.mean((td_targets - value[:-1]) ** 2)

        elif self.grpo_cfg.supervision_mode == "outcome":
            rewards = data.reward.astype(jnp.float32)  # (T,B)
            valid = data.extras.get("valid_mask", jnp.ones_like(rewards))  # (T,B)
            # 只累计有效步；若 env 在 done 步给了非零 reward，一样会被计入
            R = (rewards * valid).sum(axis=0)  # (B,)
            mu = R.mean(keepdims=True);
            sd = R.std(keepdims=True) + 1e-8
            A_traj = (R - mu) / sd  # (B,)
            advantage = jnp.broadcast_to(A_traj, rewards.shape) * valid  # (T,B)，pad处为0
            critic_loss = 0.0


        elif self.grpo_cfg.supervision_mode == "process":
            rew = data.reward.astype(jnp.float32)  # (T,B)
            r_mu = jnp.mean(rew);
            r_sd = jnp.std(rew) + 1e-8
            norm_r = (rew - r_mu) / r_sd
            rev = jnp.flip(norm_r, axis=0)
            csum_rev = jnp.cumsum(rev, axis=0)
            future_sum = jnp.flip(csum_rev, axis=0)
            future_sum_excl = jnp.concatenate([future_sum[1:], jnp.zeros_like(future_sum[:1])], axis=0)
            advantage = future_sum_excl
            critic_loss = 0.0


        else:
            raise ValueError(f"Unknown supervision_mode: {self.grpo_cfg.supervision_mode}")

        # （可选）二次标准化调试
        metrics = {}
        if getattr(self.grpo_cfg, "normalize_adv", False):
            metrics.update(unnormalized_advantage=jnp.mean(advantage))
            mu2 = advantage.mean()
            sd2 = advantage.std() + 1e-8
            jdebug.print("DBG[norm2] mu2={}, sd2={}", mu2, sd2)
            advantage = (advantage - mu2) / sd2

        # ========== 3) 当前 params 前向 → ratio / 裁剪 ==========
        policy_apply = self.actor_critic_networks.policy_network.apply

        def scan_body(_, obs_t):
            return None, policy_apply(params.actor, obs_t)

        _, new_logits = jax.lax.scan(scan_body, None, data.observation)
        dist = self.actor_critic_networks.parametric_action_distribution

        logp_new = dist.log_prob(new_logits, data.action)  # (T,B)
        ratio = jnp.exp(logp_new - data.log_prob)  # (T,B) old policy from run_epoch

        weight = data.extras.get("valid_mask", jnp.ones_like(ratio))  # (T,B)
        clipped = jnp.clip(ratio, 1.0 - self.grpo_cfg.clip_eps, 1.0 + self.grpo_cfg.clip_eps)
        pg = jnp.minimum(ratio, clipped) * jax.lax.stop_gradient(advantage)
        policy_loss = - (pg * weight).sum() / (weight.sum() + 1e-8)

        # approx_kl / entropy 也用同样的权重：
        approx_kl = ((data.log_prob - logp_new) * weight).sum() / (weight.sum() + 1e-8)

        key, ent_key = jax.random.split(acting_state.key)
        entropy_t = dist.entropy(new_logits, ent_key)
        entropy   = (entropy_t * weight).sum() / (weight.sum() + 1e-8)
        entropy_loss = -entropy
        acting_state = acting_state._replace(key=key)
        total_loss = (
                self.l_pg * policy_loss
                + self.l_td * critic_loss
                + self.l_en * entropy_loss
        )
        kl_coef = getattr(self.grpo_cfg, "kl_coef", 0.0)
        if kl_coef > 0.0:
            total_loss = total_loss + kl_coef * approx_kl


        # ========== 5) 指标 ==========
        metrics.update(
            total_loss=total_loss,
            policy_loss=policy_loss,
            critic_loss=critic_loss,
            entropy_loss=entropy_loss,
            entropy=entropy,
            advantage=jnp.mean(advantage),
            value=jnp.mean(value),
            approx_kl=approx_kl,
        )

        return total_loss, (acting_state, metrics)

    def run_epoch(self, training_state):
        params = training_state.params_state.params
        opt_state = training_state.params_state.opt_state
        update_count = training_state.params_state.update_count
        acting_state = training_state.acting_state

        acting_state, data = self.rollout_episodic(policy_params=params.actor, acting_state=acting_state)

        valid_mask = data.extras["valid_mask"]  # shape (T, B)
        done_mask = (data.discount == 0.0) & (valid_mask > 0)
        steps_per_env = valid_mask.sum(axis=0)
        avg_steps_per_env = steps_per_env.mean()
        num_episodes_per_env = done_mask.sum(axis=0)
        avg_episodes_per_env = num_episodes_per_env.mean()

        metrics = {
            "avg_steps_per_env": avg_steps_per_env,
            "avg_episodes_per_env": avg_episodes_per_env,
            "total_episodes": num_episodes_per_env.sum(),
        }

        policy_apply = self.actor_critic_networks.policy_network.apply
        dist = self.actor_critic_networks.parametric_action_distribution
        logp_old = dist.log_prob(data.extras["old_logits"], data.action)
        data = data._replace(log_prob=jax.lax.stop_gradient(logp_old))

        K = getattr(self.grpo_cfg, "num_policy_updates", 3)
        for k in range(K):
            (loss, (acting_state, inner_metrics)), grads = jax.value_and_grad(
                self._loss_from_data, has_aux=True
            )(params, acting_state, data)

            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = jax.tree_util.tree_map(lambda w, u: w + u, params, updates)

        # 合并训练指标
        metrics.update(inner_metrics)

        new_params_state = training_state.params_state._replace(
            params=params,
            opt_state=opt_state,
            update_count=update_count + K,
        )
        new_state = training_state._replace(
            params_state=new_params_state,
            acting_state=acting_state,
        )
        return new_state, metrics

    def rollout_episodic(self, policy_params, acting_state):
        """收集每个并行 actor 的完整 episode（上界 T_max），并返回带 valid_mask 的 Transition。"""
        T_max = self.n_steps  # 现在把 n_steps 当作 “每个episode的最大步数上限”
        env = self.env
        dist = self.actor_critic_networks.parametric_action_distribution
        policy_apply = self.actor_critic_networks.policy_network.apply

        key = acting_state.key
        B = self.batch_size

        key, reset_base = jax.random.split(key)
        reset_keys = jax.random.split(reset_base, B)

        # 初始：重置环境，获得第0步观测
        state, timestep = env.reset(reset_keys)
        first_obs = timestep.observation

        B = jax.tree_util.tree_leaves(first_obs)[0].shape[0]  # 并行actor数

        def step_fn(carry, _):
            key, state, obs_t, done_mask = carry
            key, subkey = jax.random.split(key)

            logits_t = policy_apply(policy_params, obs_t)
            action_t = dist.sample(logits_t, subkey)  #

            next_state, timestep = env.step(state, action_t)
            newly_done = (timestep.step_type == StepType.LAST)
            active = ~done_mask  # (B,)

            reward_t = jnp.where(active, timestep.reward, 0.0)  # (B,)
            discount_t = jnp.where(active, timestep.discount, 0.0)  # (B,)

            def select_by_active(old_leaf, new_leaf):
                m = active
                for _ in range(new_leaf.ndim - m.ndim):
                    m = m[..., None]
                return jnp.where(m, new_leaf, old_leaf)

            next_obs_t = jax.tree_util.tree_map(select_by_active, obs_t, timestep.observation)
            next_state = jax.tree_util.tree_map(select_by_active, state, next_state)

            done_mask_new = jnp.logical_or(done_mask, newly_done)
            valid_t = active.astype(jnp.float32)

            transition = (obs_t, next_obs_t, action_t, reward_t, discount_t, valid_t, logits_t)
            carry_out = (key, next_state, next_obs_t, done_mask_new)
            return carry_out, transition

        # 初始 done_mask=False
        init_carry = (key, state, first_obs, jnp.zeros((B,), dtype=bool))
        (_, _, last_obs, done_mask_final), traj = jax.lax.scan(step_fn, init_carry, None, length=T_max)

        obs_b, next_obs_b, action_b, reward_b, discount_b, valid_b, logits_b = traj  # 每项形状 (T_max, B, ...)

        # 构建 Transition（沿用 A2C 的字段），并把 valid_mask/old_logits 进 extras
        # 注意：log_prob 由 run_epoch 冻结 old policy 时再填
        data = Transition(
            observation=obs_b,
            action=action_b,
            reward=reward_b,
            discount=discount_b,
            next_observation=next_obs_b,
            log_prob=jnp.zeros_like(reward_b),  # 占位，稍后 _replace 为 old logp
            extras={"valid_mask": valid_b, "old_logits": logits_b},
            logits=logits_b
        )

        new_acting_state = acting_state._replace(key=key)
        return new_acting_state, data

