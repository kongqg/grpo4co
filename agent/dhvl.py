# dhvl_agent.py

import dataclasses

from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
import rlax
from chex import dataclass
from jumanji.training.agents.a2c import A2CAgent
from jumanji.training.types import ActingState, Transition
from jumanji.types import StepType


@dataclass
class dhvlConfig:
    clip_eps: float = 0.1
    normalize_adv: bool = True
    num_policy_updates: int = 3     # PPO epochs
    kl_coef: float = 0.0
    minibatch_size: int = 0         # 0/None => full batch

    # -----------------------------
    # Double Highway V-Learning（PPO 用）
    # -----------------------------
    use_double_highway_v: bool = True
    highway_lambda: float = 0.97
    # 丢弃最后一个未结束 episode（只影响 weight/mask；若 rollout 长度经常不足以结束 episode，请设为 False）
    drop_last_incomplete_episode: bool = False
    # Target critic EMA：只有当 params 里包含 critic_target 时才会生效
    target_ema_tau: float = 0.005


def _tree_true_like(x):
    return jax.tree.map(lambda t: jnp.ones_like(t, dtype=bool), x)


def _apply_mask_to_logits(logits, mask):
    # mask: True=valid, False=invalid
    # logits: [..., A]
    neg_inf = jnp.finfo(logits.dtype).min
    return jnp.where(mask, logits, neg_inf)


def _maybe_extract_action_mask(obs, logits):
    # 支持 obs 是 dict 且包含 action_mask
    if isinstance(obs, dict) and "action_mask" in obs:
        mask = obs["action_mask"]
        # 保证 mask 形状可以 broadcast 到 logits
        return mask.astype(bool)
    return _tree_true_like(logits)


def _take_transition_axis0(data: Transition, env_idx: jnp.ndarray, B: int) -> Transition:
    """从 Transition 的第 0 维抽取 env_idx（用于 [N,...] / [T*B,...] 的 minibatch）。"""

    def slice_leaf(t):
        if not hasattr(t, "shape"):
            return t
        # [N, ...]
        if t.ndim >= 1 and t.shape[0] == B:
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


def _params_replace(params: Any, **kwargs: Any):
    """兼容 namedtuple / flax.struct / dataclass 的 replace。"""
    if hasattr(params, "_replace"):
        return params._replace(**kwargs)
    if hasattr(params, "replace"):
        return params.replace(**kwargs)
    if dataclasses.is_dataclass(params):
        return dataclasses.replace(params, **kwargs)
    if isinstance(params, dict):
        new_p = dict(params)
        new_p.update(kwargs)
        return new_p
    raise TypeError(f"Unsupported params type for replace: {type(params)}")


# ===================================================================
# PPO Agent（保持你的 rollout/丢弃末尾不完整 episode 逻辑不变）
# ===================================================================

class dhvlAgent(A2CAgent):

    def __init__(self, dhvl_cfg: dhvlConfig, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.dhvl_cfg = dhvl_cfg
        self.batch_size = int(getattr(self, "total_batch_size", kwargs.get("total_batch_size")))

    def _loss_from_data(self, params, acting_state: ActingState, data: Transition):
        policy_apply = self.actor_critic_networks.policy_network.apply

        new_logits = policy_apply(params.actor, data.observation)

        mask_b = data.extras.get("action_mask", _tree_true_like(new_logits))
        masked_new_logits = _apply_mask_to_logits(new_logits, mask_b)

        dist = self.actor_critic_networks.parametric_action_distribution

        # 用新概率选旧动作
        logp_new = dist.log_prob(masked_new_logits, data.extras.get("raw_action"))

        ratio = jnp.exp(logp_new - data.log_prob)
        advantage = data.extras.get("adv", None)
        targets = data.extras.get("targ", None)
        weight = data.extras.get("valid_mask", jnp.ones_like(ratio, dtype=jnp.float32))
        weight = weight.astype(jnp.float32)

        clipped = jnp.clip(ratio, 1.0 - self.dhvl_cfg.clip_eps, 1.0 + self.dhvl_cfg.clip_eps)

        surr1 = ratio * jax.lax.stop_gradient(advantage)
        surr2 = clipped * jax.lax.stop_gradient(advantage)
        policy_loss = - (jnp.minimum(surr1, surr2) * weight).sum() / (weight.sum() + 1e-8)

        approx_kl = ((data.log_prob - logp_new) * weight).sum() / (weight.sum() + 1e-8)

        key, ent_key = jax.random.split(acting_state.key)
        acting_state = acting_state._replace(key=key)

        entropy_t = dist.entropy(masked_new_logits, ent_key)
        entropy = (entropy_t * weight).sum() / (weight.sum() + 1e-8)
        entropy_loss = -entropy

        # -----------------------------
        # value_loss（标准 PPO：clipped value loss）
        # -----------------------------
        vapply = self.actor_critic_networks.value_network.apply

        value_new = vapply(params.critic, data.observation).astype(jnp.float32)
        targets = jax.lax.stop_gradient(targets.astype(jnp.float32))

        value_old = data.extras.get("value_old", None)
        value_old = jax.lax.stop_gradient(value_old.astype(jnp.float32))
        v_clipped = value_old + jnp.clip(
            value_new - value_old,
            -self.dhvl_cfg.clip_eps,
            self.dhvl_cfg.clip_eps,
        )

        v_loss1 = (value_new - targets) ** 2
        v_loss2 = (v_clipped - targets) ** 2
        v_loss = jnp.maximum(v_loss1, v_loss2)
        value_loss = 0.5 * (v_loss * weight).sum() / (weight.sum() + 1e-8)

        total_loss = self.l_pg * policy_loss + self.l_td * value_loss + self.l_en * entropy_loss
        if self.dhvl_cfg.kl_coef > 0.0:
            total_loss = total_loss + self.dhvl_cfg.kl_coef * approx_kl

        metrics = dict(
            total_loss=total_loss,
            policy_loss=policy_loss,
            critic_loss=value_loss,
            entropy=entropy,
            approx_kl=approx_kl,
        )
        return total_loss, (acting_state, metrics)

    def rollout_episodic(
        self,
        policy_params,
        value_params,
        acting_state: ActingState,
        value_target_params: Optional[Any] = None,
    ):
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
                episode_count=acting_state.episode_count + next_timestep.last().sum(),
                env_step_count=acting_state.env_step_count + self.batch_size_per_device,
            )

            transition = (
                obs_t,
                action_t,
                raw_action_t,
                next_timestep.reward,
                next_timestep.discount,
                next_timestep.observation,
                logp_t,
                mask_t,
                logits_t,
                v_t,
            )
            return new_acting_state, transition

        acting_keys = jax.random.split(acting_state.key, T_max).reshape((T_max, -1))
        new_acting_state, traj = jax.lax.scan(run_one_step, acting_state, acting_keys)

        obs_b, action_b, raw_action_b, reward_b, discount_b, next_obs_b, logp_b, mask_b, logits_b, v_b = traj

        # bootstrap：与 A2C 对齐，用 rollout 最后时刻的 observation
        last_obs = new_acting_state.timestep.observation
        last_v_b = value_apply(value_params, last_obs).astype(jnp.float32)

        # ==============================================================
        # Double Highway V-Learning targets (用于 PPO: targets + advantage)
        #
        # 对应你的公式：
        # - V1 := value_params（online / choice，用于 gate 与 baseline）
        # - V2 := value_target_params（target / eval，用于构造 V_target）
        # - High_t := (λ * V_choice_{t+1} > V1(s_{t+1}))
        # - V_choice_t := r_t + γ d_t * ( High_t ? λ*V_choice_{t+1} : V1(s_{t+1}) )
        # - V_target_t := r_t + γ d_t * ( High_t ? λ*V_target_{t+1} : V2(s_{t+1}) )
        #
        # 备注：
        # - 如果你想严格复现“Double 公式里不含 λ”的版本，把 highway_lambda 设成 1.0 即可。
        # - 若 value_target_params 未提供，则退化为单网络（V2=V1）。
        # ==============================================================
        gamma = getattr(self, "discount_factor", 0.99)
        hlam = getattr(self.dhvl_cfg, "highway_lambda", 0.97)

        v1_old = v_b.astype(jnp.float32)  # [T,B] == V1(s_t)

        # 计算 V2(s_t)（若没有 target params 则直接用 V1）
        if value_target_params is None:
            v2_old = v1_old
            last_v2_b = last_v_b
        else:
            def v2_fwd(_, ob_t):
                v2_t = value_apply(value_target_params, ob_t).astype(jnp.float32)
                return None, v2_t

            _, v2_old = jax.lax.scan(v2_fwd, None, obs_b)  # [T,B]
            last_v2_b = value_apply(value_target_params, last_obs).astype(jnp.float32)

        # next-state values：V(s_{t+1})
        v1_tp1 = jnp.concatenate([v1_old[1:], last_v_b[None]], axis=0)   # [T,B] == V1(s_{t+1})
        v2_tp1 = jnp.concatenate([v2_old[1:], last_v2_b[None]], axis=0)  # [T,B] == V2(s_{t+1})

        r = reward_b.astype(jnp.float32)    # [T,B]
        d = discount_b.astype(jnp.float32)  # [T,B] 终止=0，非终止=1

        # 可选：丢弃最后一个未结束 episode（只影响 weight/mask）
        if getattr(self.dhvl_cfg, "drop_last_incomplete_episode", True):
            is_done = (d == 0.0)  # [T,B]
            done_inclusive_scan = jnp.cumsum(is_done[::-1], axis=0)[::-1]
            valid_mask = (done_inclusive_scan > 0).astype(jnp.float32)  # [T,B]
        else:
            valid_mask = jnp.ones_like(d, dtype=jnp.float32)

        def dhv_scan(state, xs):
            choice_next, target_next = state  # [B], [B]
            r_t, d_t, v1_tp1_t, v2_tp1_t = xs  # each: [B]

            high = (hlam * choice_next > v1_tp1_t)

            choice_t = r_t + gamma * d_t * jnp.where(high, hlam * choice_next, v1_tp1_t)
            target_t = r_t + gamma * d_t * jnp.where(high, hlam * target_next, v2_tp1_t)

            return (choice_t, target_t), (choice_t, target_t, high)

        init_state = (last_v_b, last_v2_b)  # (V_choice_T, V_target_T)
        _, (choice_rev, target_rev, high_rev) = jax.lax.scan(
            dhv_scan,
            init_state,
            (r[::-1], d[::-1], v1_tp1[::-1], v2_tp1[::-1]),
        )

        V_target = target_rev[::-1]  # [T,B]
        high_gate = high_rev[::-1].astype(jnp.float32)  # [T,B] for debug

        # targets 给 critic：不要归一化
        targets = jax.lax.stop_gradient(V_target) * valid_mask

        # advantage 给 actor：baseline 用 V1(s_t)
        raw_advantage = (V_target - v1_old) * valid_mask
        advantage = raw_advantage

        if self.dhvl_cfg.normalize_adv:
            den = jnp.maximum(valid_mask.sum(), 1.0)
            mu = (advantage * valid_mask).sum() / den
            var = (((advantage - mu) ** 2) * valid_mask).sum() / den
            sd = jnp.sqrt(var + 1e-8)
            advantage = ((advantage - mu) / sd) * valid_mask

        # 可选调试指标：门控比例（只在 valid 上统计）
        gate_ratio = (high_gate * valid_mask).sum() / (valid_mask.sum() + 1e-8)

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
                "valid_mask": valid_mask,
                "gate_ratio": gate_ratio,
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
            value_target_params=getattr(params, "critic_target", None),
        )

        # ---- always minibatch + shuffle ----
        minibatch_size = 64
        max_updates = 10

        # flatten [T,B] -> [T*B]
        if hasattr(data.reward, "shape") and data.reward.ndim == 2:
            B = int(data.reward.shape[1])
            data_flat = _flatten_transition_TB(data, B)
        else:
            data_flat = data

        # total N
        N = int(data_flat.reward.shape[0]) if hasattr(data_flat.reward, "shape") else 0

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

        # -----------------------------
        # 更新 target critic（EMA）
        # 只有当 params 中包含 critic_target 时才执行；否则 gate 会退化为单网络
        # -----------------------------
        if hasattr(params, "critic_target"):
            tau = getattr(self.dhvl_cfg, "target_ema_tau", 0.005)
            new_target = jax.tree.map(lambda t, s: (1.0 - tau) * t + tau * s, params.critic_target, params.critic)
            params = _params_replace(params, critic_target=new_target)

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
