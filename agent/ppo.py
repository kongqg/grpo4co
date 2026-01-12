# grpo_agent.py

from typing import Dict, Any

import jax
import jax.numpy as jnp
import rlax
from chex import dataclass
from jumanji.training.agents.a2c import A2CAgent
from jumanji.training.types import ActingState, Transition
from jumanji.types import StepType


@dataclass
class PPOConfig:
    clip_eps: float = 0.1
    normalize_adv: bool = True
    num_policy_updates: int = 3     # PPO epochs
    kl_coef: float = 0.0
    minibatch_size: int = 0         # 0/None => full batch


def _tree_true_like(x):
    return jax.tree.map(lambda t: jnp.ones_like(t, dtype=bool), x)


def _apply_mask_to_logits(logits, mask):
    def f(l, m):
        m = jnp.asarray(m, dtype=bool).reshape(l.shape)
        return jnp.where(m, l, jnp.full_like(l, -1e9))
    return jax.tree.map(f, logits, mask)


def _maybe_extract_action_mask(obs, logits):
    mask = None
    if hasattr(obs, "action_mask"):
        mask = obs.action_mask
    elif isinstance(obs, dict) and "action_mask" in obs:
        mask = obs["action_mask"]
    return jax.tree.map(lambda m: m.astype(bool), mask) if mask is not None else _tree_true_like(logits)


def _slice_transition_by_env(data: Transition, env_idx: jnp.ndarray, B: int) -> Transition:
    """沿 env/batch 维切分 Transition（[T,B,...] 切 axis=1，[B,...] 切 axis=0）。"""

    def slice_leaf(t):
        if not hasattr(t, "shape"):
            return t
        # [T, B, ...]
        if t.ndim >= 2 and t.shape[1] == B:
            return jnp.take(t, env_idx, axis=1)
        # [B, ...] 或 [B]
        if t.ndim >= 1 and t.shape[0] == B and (t.ndim == 1 or t.shape[1] != B):
            return jnp.take(t, env_idx, axis=0)
        return t

    return Transition(
        observation=jax.tree.map(slice_leaf, data.observation),
        action=jax.tree.map(slice_leaf, data.action),
        reward=slice_leaf(data.reward),
        discount=slice_leaf(data.discount),
        next_observation=jax.tree.map(slice_leaf, data.next_observation),
        log_prob=slice_leaf(data.log_prob),
        extras=jax.tree.map(slice_leaf, data.extras),
        logits=jax.tree.map(slice_leaf, data.logits),
    )

def _flatten_transition_TB(data: Transition, B: int) -> Transition:
    """把 [T,B,...] 展平成 [T*B,...]（只对前两维匹配的叶子生效）。"""
    # 以 reward 的 [T,B] 形状推断 T
    T = data.reward.shape[0] if hasattr(data.reward, "shape") and data.reward.ndim >= 2 else None

    def flatten_leaf(t):
        if not hasattr(t, "shape") or T is None:
            return t
        # [T, B, ...] -> [T*B, ...]
        if t.ndim >= 2 and t.shape[0] == T and t.shape[1] == B:
            return t.reshape((T * B,) + t.shape[2:])
        return t

    return Transition(
        observation=jax.tree.map(flatten_leaf, data.observation),
        action=jax.tree.map(flatten_leaf, data.action),
        reward=flatten_leaf(data.reward),
        discount=flatten_leaf(data.discount),
        next_observation=jax.tree.map(flatten_leaf, data.next_observation),
        log_prob=flatten_leaf(data.log_prob),
        extras=jax.tree.map(flatten_leaf, data.extras),
        logits=jax.tree.map(flatten_leaf, data.logits),
    )


def _take_transition_axis0(data: Transition, idx: jnp.ndarray, N: int) -> Transition:
    """对所有 shape[0]==N 的叶子沿 axis=0 取 idx（用于 flat 后的 mini-batch）。"""

    def take_leaf(t):
        if not hasattr(t, "shape"):
            return t
        if t.ndim >= 1 and t.shape[0] == N:
            return jnp.take(t, idx, axis=0)
        return t

    return Transition(
        observation=jax.tree.map(take_leaf, data.observation),
        action=jax.tree.map(take_leaf, data.action),
        reward=take_leaf(data.reward),
        discount=take_leaf(data.discount),
        next_observation=jax.tree.map(take_leaf, data.next_observation),
        log_prob=take_leaf(data.log_prob),
        extras=jax.tree.map(take_leaf, data.extras),
        logits=jax.tree.map(take_leaf, data.logits),
    )

# ===================================================================
# PPO Agent（保持你的 rollout/丢弃末尾不完整 episode 逻辑不变）
# ===================================================================

class PPOAgent(A2CAgent):

    def __init__(self, grpo_cfg: PPOConfig, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.grpo_cfg = grpo_cfg
        self.batch_size = int(getattr(self, "total_batch_size", kwargs.get("total_batch_size")))



    def _loss_from_data(self, params, acting_state: ActingState, data: Transition):
        policy_apply = self.actor_critic_networks.policy_network.apply

        def fwd(_, ob):
            return None, policy_apply(params.actor, ob)

        _, new_logits = jax.lax.scan(fwd, None, data.observation)

        mask_b = data.extras.get("action_mask", _tree_true_like(new_logits))
        masked_new_logits = _apply_mask_to_logits(new_logits, mask_b)

        dist = self.actor_critic_networks.parametric_action_distribution
        # 用新概率选旧动作
        logp_new = dist.log_prob(masked_new_logits, data.extras.get("raw_action"))

        ratio = jnp.exp(logp_new - data.log_prob)
        advantage = data.extras.get("adv", None)
        targets = data.extras.get("targ", None)
        # weight = data.extras.get("valid_mask", jnp.ones_like(ratio))
        weight = jnp.ones_like(ratio, dtype=jnp.float32)
        clipped = jnp.clip(ratio, 1.0 - self.grpo_cfg.clip_eps, 1.0 + self.grpo_cfg.clip_eps)

        surr1 = ratio * jax.lax.stop_gradient(advantage)
        surr2 = clipped * jax.lax.stop_gradient(advantage)
        policy_loss = - (jnp.minimum(surr1, surr2) * weight).sum() / (weight.sum() + 1e-8)

        approx_kl = ((data.log_prob - logp_new) * weight).sum() / (weight.sum() + 1e-8)

        key, ent_key = jax.random.split(acting_state.key)
        entropy_t = dist.entropy(masked_new_logits, ent_key)
        entropy = (entropy_t * weight).sum() / (weight.sum() + 1e-8)
        entropy_loss = -entropy

        # -----------------------------
        # value_loss（标准 PPO：clipped value loss）
        # -----------------------------
        vapply = self.actor_critic_networks.value_network.apply

        def v_fwd(_, ob_t):
            v_new_t = vapply(params.critic, ob_t).astype(jnp.float32)
            return None, v_new_t

        _, value_new = jax.lax.scan(v_fwd, None, data.observation)  # [T, B]
        targets = jax.lax.stop_gradient(targets.astype(jnp.float32))  # [T, B]

        value_old = data.extras.get("value_old", None)
        value_old = jax.lax.stop_gradient(value_old.astype(jnp.float32))  # [T, B]
        v_clipped = value_old + jnp.clip(
            value_new - value_old,
            -self.grpo_cfg.clip_eps,
            self.grpo_cfg.clip_eps,
        )
        v_loss1 = (value_new - targets) ** 2
        v_loss2 = (v_clipped - targets) ** 2
        v_loss = jnp.maximum(v_loss1, v_loss2)
        value_loss = 0.5 * (v_loss * weight).sum() / (weight.sum() + 1e-8)


        total_loss = self.l_pg * policy_loss + self.l_td * value_loss + self.l_en * entropy_loss
        if self.grpo_cfg.kl_coef > 0.0:
            total_loss = total_loss + self.grpo_cfg.kl_coef * approx_kl

        metrics = dict(
            total_loss=total_loss,
            policy_loss=policy_loss,
            critic_loss=value_loss,
            entropy_loss=entropy_loss,
            entropy=entropy,
            advantage=advantage.mean(),
            value=value_new,
            approx_kl=approx_kl,
        )
        acting_state = acting_state._replace(key=key)
        return total_loss, (acting_state, metrics)

    def rollout_episodic(self, policy_params, value_params, acting_state: ActingState):
        T_max = self.n_steps
        env = self.env
        dist = self.actor_critic_networks.parametric_action_distribution
        policy_apply = self.actor_critic_networks.policy_network.apply
        value_apply = self.actor_critic_networks.value_network.apply

        def run_one_step(acting_state: ActingState, key):
            timestep = acting_state.timestep
            obs_t = timestep.observation

            logits_t = policy_apply(policy_params, obs_t)
            mask_t = _maybe_extract_action_mask(obs_t, logits_t)
            masked_logits_t = _apply_mask_to_logits(logits_t, mask_t)

            raw_action_t = dist.sample_no_postprocessing(masked_logits_t, key)
            logp_t = dist.log_prob(masked_logits_t, raw_action_t)
            action_t = dist.postprocess(raw_action_t)

            # PPO 需要 v_old：在 rollout 里存下来
            v_t = value_apply(value_params, obs_t).astype(jnp.float32)

            next_state, next_timestep = env.step(acting_state.state, action_t)

            new_acting_state = ActingState(
                state=next_state,
                timestep=next_timestep,
                key=key,
                episode_count=acting_state.episode_count
                              + jax.lax.psum(next_timestep.last().sum(), "devices"),
                env_step_count=acting_state.env_step_count
                               + jax.lax.psum(self.batch_size_per_device, "devices"),
            )

            transition = (
                obs_t, action_t, raw_action_t,
                next_timestep.reward, next_timestep.discount, next_timestep.observation,
                logp_t, mask_t, logits_t, v_t
            )
            return new_acting_state, transition

        acting_keys = jax.random.split(acting_state.key, T_max).reshape((T_max, -1))
        new_acting_state, traj = jax.lax.scan(run_one_step, acting_state, acting_keys)

        obs_b, action_b, raw_action_b, reward_b, discount_b, next_obs_b, logp_b, mask_b, logits_b, v_b = traj

        # bootstrap：与 A2C 对齐，用 rollout 最后时刻的 observation
        last_obs = new_acting_state.timestep.observation
        last_v_b = value_apply(value_params, last_obs).astype(jnp.float32)

        # ===== 下面 GAE / targets 可以保持你原来的写法 =====
        gamma = getattr(self, "discount_factor", 0.99)
        lam = getattr(self, "bootstrapping_factor", 0.95)

        v_old = v_b.astype(jnp.float32)  # [T,B]
        v_tp1 = jnp.concatenate([v_old[1:], last_v_b[None]], axis=0)
        r = reward_b.astype(jnp.float32)
        d = discount_b.astype(jnp.float32)  # 终止=0，非终止=1（若不是，先打印 min/max）

        delta = r + gamma * d * v_tp1 - v_old

        def gae_scan(gae, xs):
            delta_t, d_t = xs
            gae = delta_t + gamma * lam * d_t * gae
            return gae, gae

        _, adv_rev = jax.lax.scan(gae_scan, jnp.zeros_like(last_v_b), (delta[::-1], d[::-1]))
        advantage = adv_rev[::-1]

        if self.grpo_cfg.normalize_adv:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        targets = jax.lax.stop_gradient(advantage + v_old)

        data = Transition(
            observation=obs_b,
            action=action_b,
            reward=reward_b,
            discount=discount_b,
            next_observation=next_obs_b,
            log_prob=logp_b,
            extras={
                "action_mask": mask_b,
                "value_old": v_b,
                "last_value_old": last_v_b,
                "adv": advantage,
                "targ": targets,
                "raw_action": raw_action_b,
            },
            logits=logits_b,
        )
        return new_acting_state, data

    def run_epoch(self, training_state):
        params = training_state.params_state.params
        opt_state = training_state.params_state.opt_state
        update_count = training_state.params_state.update_count
        acting_state = training_state.acting_state

        # 1) rollout
        acting_state, data = self.rollout_episodic(
            policy_params=params.actor,
            value_params=params.critic,
            acting_state=acting_state,
        )

        # ---- always minibatch + shuffle ----
        minibatch_size = 64
        max_updates = 10

        N = int(data.reward.size)  # [T,B] => T*B

        # flatten: [T,B,...] -> [N,...]
        if hasattr(data.reward, "ndim") and data.reward.ndim >= 2:
            B = int(data.reward.shape[1])
            data_flat = _flatten_transition_TB(data, B)
        else:
            data_flat = data

        # shuffle indices
        key, perm_key = jax.random.split(acting_state.key)
        acting_state = acting_state._replace(key=key)
        perm = jax.random.permutation(perm_key, N)

        # how many minibatches exist (ceil)
        num_batches = (N + minibatch_size - 1) // minibatch_size
        num_updates = min(max_updates, num_batches)  # <=10，且最后一批不满64也会被算进来

        metrics = {}

        for i in range(num_updates):
            start = i * minibatch_size
            end = min((i + 1) * minibatch_size, N)  # 关键：最后一批不足64也更新
            mb_idx = perm[start:end]  # shape [<=64]

            batch_data = _take_transition_axis0(data_flat, mb_idx, N)

            (loss, (acting_state, inner_metrics)), grads = jax.value_and_grad(
                self._loss_from_data, has_aux=True
            )(params, acting_state, batch_data)

            # L2 norm
            actor_grads_leaves = jax.tree_util.tree_leaves(grads.actor)
            actor_global_norm = jnp.sqrt(
                sum(jnp.sum(jnp.square(x)) for x in actor_grads_leaves if x is not None)
            )
            inner_metrics['grad_norm/actor_global_norm'] = actor_global_norm

            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = jax.tree.map(lambda w, u: w + u, params, updates)
            metrics = inner_metrics

        new_params_state = training_state.params_state._replace(
            params=params,
            opt_state=opt_state,
            update_count=update_count + num_updates,
        )
        new_state = training_state._replace(
            params_state=new_params_state,
            acting_state=acting_state,
        )
        return new_state, metrics
