# grpo_agent.py

from typing import Dict, Tuple, Any

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
    clip_eps: float = 0.1
    normalize_adv: bool = True
    supervision_mode: str = "outcome"
    num_policy_updates: int = 3
    kl_coef: float = 0.0
    repeats_per_instance: int = 1

def _tree_true_like(x):
    return jax.tree_map(lambda t: jnp.ones_like(t, dtype=bool), x)


def _apply_mask_to_logits(logits, mask):
    return jax.tree_map(lambda l, m: l + jnp.where(m, 0.0, -1e9).astype(l.dtype), logits, mask)


def _maybe_extract_action_mask(obs, logits):
    mask = None
    if hasattr(obs, "action_mask"):
        mask = obs.action_mask
    elif isinstance(obs, dict) and "action_mask" in obs:
        mask = obs["action_mask"]
    return jax.tree_map(lambda m: m.astype(bool), mask) if mask is not None else _tree_true_like(logits)

# ===================================================================
# GRPO Agent 实现
# ===================================================================

class GRPOAgent(A2CAgent):

    def __init__(self, grpo_cfg: GRPOConfig, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.grpo_cfg = grpo_cfg
        self.batch_size = int(getattr(self, "total_batch_size", kwargs.get("total_batch_size")))

    def _compute_advantage(self, params, data: Transition):
        if self.grpo_cfg.supervision_mode == "outcome":
            r = data.reward.astype(jnp.float32) # (T, B)
            valid = data.extras.get("valid_mask", jnp.ones_like(r))
            G = int(getattr(self.grpo_cfg, "repeats_per_instance", 8))
            T, B = r.shape
            U = B // G
            R = (r * valid).sum(axis=0)  # (B,)
            R = R.reshape(U, G)
            if self.grpo_cfg.normalize_adv:
                mu = R.mean(axis=1, keepdims=True) # (U, 1)
                sd = R.std(axis=1, keepdims=True)
                eps = 1e-8
                sd = jnp.maximum(sd, eps)
                R = (R - mu) / sd
            R = R.reshape(B, )
            adv = jnp.broadcast_to(R, (T, B)) * valid
            critic_loss = jnp.array(0.0, dtype=jnp.float32)
            return adv, 0.0
        elif self.grpo_cfg.supervision_mode == "ppo_by_steps":
            advantage = data.extras.get("adv", None)
            targets = data.extras.get("targ", None)
            assert (advantage is not None) and (targets is not None), "adv/targ must be precomputed in rollout."
            return advantage, targets
        else:
            raise ValueError(f"Unknown supervision_mode: {self.grpo_cfg.supervision_mode}")

    def _loss_from_data(self, params, acting_state: ActingState, data: Transition):
        policy_apply = self.actor_critic_networks.policy_network.apply

        def fwd(_, ob):
            return None, policy_apply(params.actor, ob)

        _, new_logits = jax.lax.scan(fwd, None, data.observation)
        mask_b = data.extras.get("action_mask", _tree_true_like(new_logits))
        masked_new_logits = _apply_mask_to_logits(new_logits, mask_b)
        dist = self.actor_critic_networks.parametric_action_distribution
        logp_new = dist.log_prob(masked_new_logits, data.action)
        ratio = jnp.exp(logp_new - data.log_prob)
        advantage, targets = self._compute_advantage(params, data)
        weight = data.extras.get("valid_mask", jnp.ones_like(ratio))
        clipped = jnp.clip(ratio, 1.0 - self.grpo_cfg.clip_eps, 1.0 + self.grpo_cfg.clip_eps)

        surr1 = ratio * jax.lax.stop_gradient(advantage)
        surr2 = clipped * jax.lax.stop_gradient(advantage)
        policy_loss = - (jnp.minimum(surr1, surr2) * weight).sum() / (weight.sum() + 1e-8)


        approx_kl = ((data.log_prob - logp_new) * weight).sum() / (weight.sum() + 1e-8)
        key, ent_key = jax.random.split(acting_state.key)
        entropy_t = dist.entropy(masked_new_logits, ent_key)
        entropy = (entropy_t * weight).sum() / (weight.sum() + 1e-8)
        entropy_loss = -entropy

        # value_loss
        if self.grpo_cfg.supervision_mode == "ppo_by_steps":
            vapply = self.actor_critic_networks.value_network.apply
            def v_fwd(_, ob_t):
                v_new_t = vapply(params.critic, ob_t).astype(jnp.float32)
                return None, v_new_t

            _, value_new = jax.lax.scan(v_fwd, None, data.observation)  # [T, B]
            targets = jax.lax.stop_gradient(targets.astype(jnp.float32))  # [T, B]
            val_err = (value_new - targets) ** 2
            value_loss = 0.5 * (val_err * weight).sum() / (weight.sum() + 1e-8)
        elif self.grpo_cfg.supervision_mode == "grpo":
            value_loss = 0.0
            value_new = 0.0
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
        key = acting_state.key
        if self.grpo_cfg.supervision_mode == "outcome":
            G = int(getattr(self.grpo_cfg, "repeats_per_instance", 1))
            B = self.batch_size
            U = B // G
            key, reset_base = jax.random.split(key)
            base_keys  = jax.random.split(reset_base, U)
            reset_keys = jnp.repeat(base_keys, repeats=G, axis=0)
            state, timestep = env.reset(reset_keys)
        elif self.grpo_cfg.supervision_mode == "ppo_by_steps":
            B = self.batch_size
            key, reset_base = jax.random.split(key)
            base_keys = jax.random.split(reset_base, B)
            state, timestep = env.reset(base_keys)
        first_obs = timestep.observation
        B = jax.tree_leaves(first_obs)[0].shape[0]
        def step_fn(carry, _):
            key, state, obs_t, done_mask = carry # key: shape(2,)
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, B)  # (B,)
            env_ids = jnp.arange(B, dtype=jnp.uint32)
            subkeys = jax.vmap(lambda k, i: jax.random.fold_in(k, i))(subkeys, env_ids)  # (B, 2)

            logits_t = policy_apply(policy_params, obs_t)
            mask_t = _maybe_extract_action_mask(obs_t, logits_t)
            masked_logits_t = _apply_mask_to_logits(logits_t, mask_t)

            if self.grpo_cfg.supervision_mode == "ppo_by_steps":
                value_apply = self.actor_critic_networks.value_network.apply
                v_t = value_apply(value_params, obs_t).astype(jnp.float32)
            elif self.grpo_cfg.supervision_mode == "grpo":
                v_t = 0.0
            action_t = jax.vmap(dist.sample, in_axes=(0, 0))(masked_logits_t, subkeys)
            logp_t = dist.log_prob(masked_logits_t, action_t)
            next_state, timestep = env.step(state, action_t)
            newly_done = (timestep.step_type == StepType.LAST)
            active = ~done_mask
            reward_t = jnp.where(active, timestep.reward, 0.0)
            discount_t = jnp.where(active, timestep.discount, 0.0)
            def sel(old_leaf, new_leaf):
                m = active
                for _ in range(new_leaf.ndim - m.ndim):
                    m = m[..., None]
                return jnp.where(m, new_leaf, old_leaf)
            next_obs_t = jax.tree_map(sel, obs_t, timestep.observation)
            next_state = jax.tree_map(sel, state, next_state)
            done_mask_new = jnp.logical_or(done_mask, newly_done)
            valid_t = active.astype(jnp.float32)
            transition = (obs_t, action_t, reward_t, discount_t, next_obs_t, valid_t, logp_t, mask_t, logits_t, v_t)
            carry_out = (key, next_state, next_obs_t, done_mask_new)
            return carry_out, transition
        init_carry = (key, state, first_obs, jnp.zeros((B,), dtype=bool))
        (_, _, last_obs, _), traj = jax.lax.scan(step_fn, init_carry, None, length=T_max)
        obs_b, action_b, reward_b, discount_b, next_obs_b, valid_b, logp_b, mask_b, logits_t, v_b = traj

        if self.grpo_cfg.supervision_mode == "ppo_by_steps":
            value_apply = self.actor_critic_networks.value_network.apply
            last_v_b = value_apply(value_params, last_obs).astype(jnp.float32)
            gamma = getattr(self, "gamma", getattr(self, "discount_factor", 0.99))
            lam = getattr(self.grpo_cfg, "lambda", 0.95)

            v_old = v_b.astype(jnp.float32)  # [T,B]
            v_tp1 = jnp.concatenate([v_old[1:], last_v_b[None, ...]], axis=0)  # [T,B]
            r = reward_b.astype(jnp.float32)
            d = discount_b.astype(jnp.float32)
            m = valid_b.astype(jnp.float32)  # 作为权重/掩码

            delta = (r + gamma * d * v_tp1 - v_old) * m
            d_eff = d * m

            def gae_scan(gae, xs):
                delta_t, d_t = xs
                gae = delta_t + gamma * lam * d_t * gae
                return gae, gae

            _, adv_rev = jax.lax.scan(gae_scan, jnp.zeros_like(last_v_b), (delta[::-1], d_eff[::-1]))
            advantage = adv_rev[::-1]  # [T,B]

            if self.grpo_cfg.normalize_adv:
                mu = (advantage * m).sum() / (m.sum() + 1e-8)
                sd = jnp.sqrt((((advantage - mu) ** 2) * m).sum() / (m.sum() + 1e-8))
                advantage = (advantage - mu) / (sd + 1e-8)

            targets = jax.lax.stop_gradient(advantage + v_old)  # [T,B]
        else:
            last_v_b = jnp.zeros((B,), dtype=jnp.float32)
            advantage = jnp.zeros_like(reward_b, dtype=jnp.float32)
            targets = jnp.zeros_like(reward_b, dtype=jnp.float32)

        data = Transition(
            observation=obs_b,
            action=action_b,
            reward=reward_b,
            discount=discount_b,
            next_observation=next_obs_b,
            log_prob=logp_b,
            extras={
                "valid_mask": valid_b,
                "action_mask": mask_b,
                "value_old": v_b,  # [T,B]
                "last_value_old": last_v_b,  # [B]
                "adv": advantage,  # [T,B]
                "targ": targets,  # [T,B]
            },
            logits=logits_t,
        )
        new_acting_state = acting_state._replace(key=key)
        return new_acting_state, data

    def run_epoch(self, training_state):
        params = training_state.params_state.params
        opt_state = training_state.params_state.opt_state
        update_count = training_state.params_state.update_count
        acting_state = training_state.acting_state
        acting_state, data = self.rollout_episodic(policy_params=params.actor,
                                                   value_params=params.critic,
                                                   acting_state=acting_state)
        K = int(getattr(self.grpo_cfg, "num_policy_updates", 3))
        metrics = {}
        for _ in range(K):
            (loss, (acting_state, inner_metrics)), grads = jax.value_and_grad(self._loss_from_data, has_aux=True)(
                params, acting_state, data
            )
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = jax.tree_map(lambda w, u: w + u, params, updates)
            metrics = inner_metrics
        new_params_state = training_state.params_state._replace(
            params=params, opt_state=opt_state, update_count=update_count + K
        )
        new_state = training_state._replace(params_state=new_params_state, acting_state=acting_state)
        return new_state, metrics

