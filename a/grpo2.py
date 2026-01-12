# ppo_by_steps.py
import sys
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
    supervision_mode: str = "outcome"
    num_policy_updates: int = 3
    kl_coef: float = 0.0
    kth_largest4percentile_method: int = 2
    mean_reward_method: bool = False

def _tree_true_like(x):
    return jax.tree.map(lambda t: jnp.ones_like(t, dtype=bool), x)


def _apply_mask_to_logits(logits, mask):
    def f(l, m):
        x = l + jnp.where(m, 0.0, -1e9).astype(l.dtype)
        x = x.astype(jnp.float32)
        x = jnp.where(x < -1e9, -1e9, x)
        return x
    return jax.tree.map(f, logits, mask)


def _maybe_extract_action_mask(obs, logits):
    mask = None
    if hasattr(obs, "action_mask"):
        mask = obs.action_mask
    elif isinstance(obs, dict) and "action_mask" in obs:
        mask = obs["action_mask"]
    return jax.tree.map(lambda m: m.astype(bool), mask) if mask is not None else _tree_true_like(logits)

# ===================================================================
# GRPO Agent 实现
# ===================================================================

class GRPOAgent(A2CAgent):

    def __init__(self, grpo_cfg: GRPOConfig, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.grpo_cfg = grpo_cfg
        self.batch_size = int(getattr(self, "total_batch_size", kwargs.get("total_batch_size")))
    def _compute_advantage(self, params, data: Transition):
        r = data.reward.astype(jnp.float32)
        done = data.extras["done"].astype(bool)
        T, B = r.shape
        prev_done = jnp.roll(done, shift=1, axis=0).at[0, :].set(False)
        ep_ids = jnp.cumsum(prev_done.astype(jnp.int32), axis=0)  # (T, B)
        num_ep = jnp.sum(done, axis=0).astype(jnp.int32)  # (B,)
        valid = (ep_ids < num_ep[None, :])  # (T, B)
        rb = jnp.swapaxes(r, 0, 1)  # (B, T)
        idsb = jnp.swapaxes(ep_ids, 0, 1)  # (B, T)
        vb = jnp.swapaxes(valid, 0, 1)  # (B, T)
        percentile = None

        def per_batch(r_b, id_b, v_b, n_b):
            data_to_sum = (r_b * v_b.astype(jnp.float32))  # (T)
            ids = jnp.where(v_b, id_b, 0)  # (T,)
            all_ids = jnp.arange(T, dtype=jnp.int32)

            def _get_sum_for_one_id(i):
                return jnp.sum(data_to_sum * (ids == i))

            seg = jax.vmap(_get_sum_for_one_id)(all_ids)  # (T,)
            seg = jnp.where(all_ids < n_b, seg, 0.0)
            return seg[ids] * v_b.astype(jnp.float32), seg

        R_step_t, seg = jax.vmap(per_batch, in_axes=(0, 0, 0, 0))(rb, idsb, vb, num_ep)
        if self.grpo_cfg.kth_largest4percentile_method > 0:
            min_value = seg.min(axis=1)

            def get_uniques(row, min_value):
                return jnp.unique(row, size=T, fill_value=min_value - 1)

            unique_values = jax.vmap(get_uniques, in_axes=(0, 0))(seg, min_value)  # batch_size, T
            sorted_value = jnp.sort(unique_values, axis=1, descending=True)  # batch_size, T
            max_value = sorted_value[..., 0]
            k = self.grpo_cfg.kth_largest4percentile_method
            # 如果选择的这个中位数是填充值，说明episode不够
            percentile = jnp.where(sorted_value[:, k - 1] == (min_value - 1), max_value,
                                   sorted_value[:, k - 1])  # Batch_size
            percentile = jnp.expand_dims(percentile, axis=0)  # 1, B
        advantage = jnp.swapaxes(R_step_t, 0, 1)  # (T, B)
        if self.grpo_cfg.mean_reward_method:
            mask = valid.astype(jnp.float32)
            count = jnp.maximum(mask.sum(), 1.0)
            mean = (advantage * mask).sum() / count
            var = (((advantage - mean) * mask) ** 2).sum() / count
            std = jnp.sqrt(var + 1e-8)
            advantage = ((advantage - mean) / std) * mask
            # jax.debug.print("adv:{x}",x=advantage)
        elif self.grpo_cfg.kth_largest4percentile_method > 0:
            mask = valid.astype(jnp.float32)
            count = jnp.maximum(mask.sum(), 1.0)
            mean = (advantage * mask).sum() / count
            var = (((advantage - mean) * mask) ** 2).sum() / count
            std = jnp.sqrt(var + 1e-8)
            advantage = ((advantage - percentile) / std) * mask

        targets = jnp.zeros_like(r)
        return advantage, targets

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
        advantage, targets = self._compute_advantage(params, data) # T, B
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
        total_loss = self.l_pg * policy_loss + self.l_en * entropy_loss
        if self.grpo_cfg.kl_coef > 0.0:
            total_loss = total_loss + self.grpo_cfg.kl_coef * approx_kl
        metrics = dict(
            total_loss=total_loss,
            policy_loss=policy_loss,
            entropy_loss=entropy_loss,
            entropy=entropy,
            advantage=advantage.mean(),
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
        B = self.batch_size
        env_id = jnp.arange(B, dtype=jnp.int32)
        key, reset_base = jax.random.split(key)
        reset_keys = jax.random.split(reset_base, B)
        reset_keys = jax.vmap(jax.random.fold_in, in_axes=(0, 0))(reset_keys, env_id)
        state, timestep = env.reset(reset_keys)
        inst_keys = reset_keys
        first_obs = timestep.observation


        def step_fn(carry, _):
            key, state, obs_t = carry
            key, actor_key = jax.random.split(key, 2)
            actor_keys = jax.vmap(lambda k, i: jax.random.fold_in(k, i))(jax.random.split(actor_key, B), env_id) # (batch_size, 1)

            logits_t = policy_apply(policy_params, obs_t)

            mask_t = _maybe_extract_action_mask(obs_t, logits_t)
            masked_logits_t = _apply_mask_to_logits(logits_t, mask_t)
            action_t = jax.vmap(dist.sample, in_axes=(0, 0))(masked_logits_t, actor_keys)

            logp_t = dist.log_prob(masked_logits_t, action_t)
            next_state_raw, timestep = env.step(state, action_t)
            reset_state, reset_timestep = env.reset(inst_keys)

            def select(mask, old_leaf, new_leaf):
                m = mask
                for _ in range(new_leaf.ndim - m.ndim):
                    m = m[..., None]
                return jnp.where(m, new_leaf, old_leaf)

            newly_done = (timestep.step_type == StepType.LAST)
            next_obs_t = jax.tree.map(lambda old, new: select(newly_done, old, new), timestep.observation,
                                      reset_timestep.observation)
            next_state = jax.tree.map(lambda old, new: select(newly_done, old, new), next_state_raw, reset_state)
            transition = (
            obs_t, action_t, timestep.reward, timestep.discount, next_obs_t, logp_t, mask_t, logits_t, newly_done)
            carry_out = (key, next_state, next_obs_t)
            return carry_out, transition

        init_carry = (key, state, first_obs)
        (_, _, _), traj = jax.lax.scan(step_fn, init_carry, None, length=T_max)
        obs_b, action_b, reward_b, discount_b, next_obs_b, logp_b, mask_b, logits_t, dones_b = traj

        data = Transition(
            observation=obs_b,
            action=action_b,
            reward=reward_b,
            discount=discount_b,
            log_prob=logp_b,
            next_observation=next_obs_b,
            extras={
                "reset_keys": inst_keys,
                "env_id": env_id,
                "action_mask": mask_b,
                "done": dones_b,
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
            params = jax.tree.map(lambda w, u: w + u, params, updates)
            metrics = inner_metrics
        new_params_state = training_state.params_state._replace(
            params=params, opt_state=opt_state, update_count=update_count + K
        )

        new_state = training_state._replace(params_state=new_params_state, acting_state=acting_state)
        return new_state, metrics
