"""
dhvl.py

A readable, standard dhvl implementation for Jumanji-style agents (JAX + Haiku),
written to be JIT/scan-friendly.

Key design points (important for correctness in JAX):
- Rollout data is collected as [T, B, ...] (time-major).
- GAE-Lambda advantages/returns are computed with a reverse lax.scan.
- Training uses flattened [N, ...] data where N = T * B.
- Minibatching uses permutation + padding + fixed-shape [num_minibatches, minibatch_size] indices,
  plus a mask to ignore padded elements. This avoids dynamic shapes in JIT/scan.

Constraint from user: `run_epoch` function name must remain unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from chex import dataclass

from jumanji.training.agents.a2c import A2CAgent
from jumanji.training.types import ActingState


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class dhvlConfig:
    clip_eps: float = 0.2
    # Value function loss weight
    vf_coef: float = 0.5
    # Entropy bonus weight
    ent_coef: float = 0.01
    # Number of dhvl epochs per rollout
    update_epochs: int = 4
    # Minibatch size used for dhvl updates
    minibatch_size: int = 64
    # Whether to normalize advantages across the rollout batch (recommended)
    normalize_adv: bool = True
    # If True, use clipped value loss (recommended for dhvl)
    clip_value_loss: bool = True
    # Use separate value clip epsilon; if None, reuse clip_eps
    value_clip_eps: float | None = None
    kl_coef: float = 0.2
    policy_delay: int = 2
    bootstrapping_factor: float = 1.0

# -----------------------------------------------------------------------------
# Internal batch structures (PyTrees)
# -----------------------------------------------------------------------------

@dataclass
class RolloutTB:
    """Time-major rollout: leading dims [T, B]."""
    observation: Any                  # pytree with leaves [T, B, ...]
    raw_action: Any                   # pytree with leaves [T, B, ...] or [T, B]
    logp_old: jnp.ndarray             # [T, B]
    value_choice_old: jnp.ndarray            # [T, B]
    value_target_old: jnp.ndarray  # [T, B]
    reward: jnp.ndarray               # [T, B]
    discount: jnp.ndarray             # [T, B]  (0 at terminal, 1 otherwise)


@dataclass
class dhvlBatchN:
    """Flattened rollout: leading dim [N] where N = T*B."""
    observation: Any            # pytree leaves [N, ...]
    raw_action: Any             # pytree leaves [N, ...] or [N]
    logp_old: jnp.ndarray       # [N]
    value_choice_old: jnp.ndarray   # [N]
    value_target_old: jnp.ndarray   # [N]
    returns_choice: jnp.ndarray     # [N]
    returns_target: jnp.ndarray     # [N]
    advantage: jnp.ndarray          # [N]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _maybe_squeeze_last(x: jnp.ndarray) -> jnp.ndarray:
    """Squeeze trailing singleton dimension if present (e.g., [B,1] -> [B])."""
    if hasattr(x, "ndim") and x.ndim > 0 and x.shape[-1] == 1:
        return jnp.squeeze(x, axis=-1)
    return x


def _extract_action_mask(observation: Any) -> Any | None:
    """Try to extract action mask from a Jumanji observation PyTree."""
    # common: dataclass/NamedTuple attribute
    if hasattr(observation, "action_mask"):
        return getattr(observation, "action_mask")
    # dict observation
    if isinstance(observation, dict) and "action_mask" in observation:
        return observation["action_mask"]
    return None


def _apply_action_mask_to_logits(logits: Any, mask: Any | None) -> Any:
    """Apply a boolean action mask to logits-like arrays (tree-friendly).

    For discrete actions: masked positions are set to a large negative number.
    If `mask` is None, logits are returned unchanged.
    """
    if mask is None:
        return logits

    def apply_one(logit_leaf, mask_leaf):
        if not isinstance(logit_leaf, jnp.ndarray) or not isinstance(mask_leaf, jnp.ndarray):
            return logit_leaf
        if mask_leaf.dtype != jnp.bool_:
            # treat non-bool masks as "no mask"
            return logit_leaf

        m = mask_leaf
        # Expand mask with trailing singleton dims until ranks match
        while m.ndim < logit_leaf.ndim:
            m = m[..., None]
        neg = jnp.array(-1e30, dtype=logit_leaf.dtype)
        return jnp.where(m, logit_leaf, neg)

    # If logits is a pytree, apply mask to every leaf; if mask isn't a pytree,
    # broadcast it to every leaf.
    if isinstance(logits, jnp.ndarray):
        return apply_one(logits, mask)

    if isinstance(mask, jnp.ndarray):
        return jax.tree_util.tree_map(lambda l: apply_one(l, mask), logits)
    return jax.tree_util.tree_map(apply_one, logits, mask)


def _tree_reshape_TB_to_N(tree: Any, T: int, B: int) -> Any:
    """Reshape leaves [T,B,...] -> [T*B,...] where applicable."""
    N = T * B

    def reshape_leaf(x):
        if not isinstance(x, jnp.ndarray):
            return x
        if x.ndim >= 2 and x.shape[0] == T and x.shape[1] == B:
            return x.reshape((N,) + x.shape[2:])
        return x

    return jax.tree_util.tree_map(reshape_leaf, tree)


def _masked_mean(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Compute mean of x over axis=0 with a {0,1} mask of same leading shape."""
    mask = mask.astype(x.dtype)
    denom = jnp.maximum(mask.sum(), 1.0)
    return (x * mask).sum() / denom


def _make_minibatch_indices(
    key: jax.Array, N: int, minibatch_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (idx, mb_mask):
    - idx: [num_minibatches, minibatch_size] with padded indices
    - mb_mask: [num_minibatches, minibatch_size] float32 mask for valid elements
    """
    num_minibatches = (N + minibatch_size - 1) // minibatch_size
    total = num_minibatches * minibatch_size

    perm = jax.random.permutation(key, N)  # [N]
    pad = total - N
    # Pad by wrapping around. Padded entries are masked out, so duplicates won't affect training.
    perm_padded = jnp.concatenate([perm, perm[:pad]], axis=0) if pad > 0 else perm
    idx = perm_padded.reshape((num_minibatches, minibatch_size))

    flat_mask = (jnp.arange(total) < N).astype(jnp.float32)
    mb_mask = flat_mask.reshape((num_minibatches, minibatch_size))
    return idx, mb_mask


def _compute_highway_v_learning(
    rewards: jnp.ndarray,          # [T,B]
    discounts: jnp.ndarray,        # [T,B]  终止=0, 非终止=1
    values_choice_old: jnp.ndarray,# [T,B]
    values_target_old: jnp.ndarray,# [T,B]
    last_value_choice: jnp.ndarray,# [B]
    last_value_target: jnp.ndarray,# [B]
    gamma: float,
    lam: float,                    #
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Highway-return version (time-major).
    Returns: (advantage, returns_choice, returns_target)
      - returns_choice: [T,B]
      - returns_target: [T,B]
      - advantage = returns_target - values_target_old
    """
    v_old_choice = values_choice_old.astype(jnp.float32)  # [T,B]
    v_tp1 = jnp.concatenate([v_old_choice[1:], last_value_choice[None, ...]], axis=0)  # [T,B]

    v_old_target = values_target_old.astype(jnp.float32)  # [T,B]
    v_tp2 = jnp.concatenate([v_old_target[1:], last_value_target[None, ...]], axis=0)  # [T,B]

    r = rewards.astype(jnp.float32)       # [T,B]
    d = discounts.astype(jnp.float32)     # [T,B]

    lambda_h = lam  # keep identical to your snippet

    def highway_step(carry, xs):
        R_next_1, R_next_2 = carry                # each [B]
        r_t, d_t, v_tp1_t, v_tp2_t = xs           # each [B]

        v_choice = lambda_h * R_next_1
        v_target = lambda_h * R_next_2

        high = (v_choice > v_tp1_t)
        v_1 = jnp.where(high, v_choice, v_tp1_t)
        v_2 = jnp.where(high, v_target, v_tp2_t)

        R_t_1 = r_t + gamma * d_t * v_1
        R_t_2 = r_t + gamma * d_t * v_2
        return (R_t_1, R_t_2), (R_t_1, R_t_2)

    _, (R_t_1, R_t_2) = jax.lax.scan(
        highway_step,
        (last_value_choice, last_value_target),
        (r[::-1], d[::-1], v_tp1[::-1], v_tp2[::-1]),
    )

    returns_choice = R_t_1[::-1]  # [T,B]
    returns_target = R_t_2[::-1]  # [T,B]

    advantage = returns_target - v_old_target
    advantage = jax.lax.stop_gradient(advantage)

    return advantage, returns_choice, returns_target


# -----------------------------------------------------------------------------
# dhvl Agent
# -----------------------------------------------------------------------------

class dhvlAgent(A2CAgent):
    """A2CAgent-compatible dhvl agent with standard GAE + minibatch shuffle."""

    def __init__(self, dhvl_cfg: dhvlConfig, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.dhvl_cfg = dhvl_cfg

    # -----------------------------
    # Rollout (collect [T,B,...])
    # -----------------------------
    def _rollout(self, params, acting_state: ActingState):
        env = self.env
        dist = self.actor_critic_networks.parametric_action_distribution
        policy_apply = self.actor_critic_networks.policy_network.apply
        value_apply = self.actor_critic_networks.value_network.apply

        T = int(self.n_steps)

        def one_step(carry: ActingState, key: jax.Array):
            ts = carry.timestep
            obs = ts.observation

            logits = policy_apply(params.actor, obs)
            mask = _extract_action_mask(obs)
            logits = _apply_action_mask_to_logits(logits, mask)

            raw_action = dist.sample_no_postprocessing(logits, key)
            logp = dist.log_prob(logits, raw_action)
            action = dist.postprocess(raw_action)

            v_t_choice, v_t_target = value_apply(params.critic, obs)
            v_t_choice = _maybe_squeeze_last(v_t_choice).astype(jnp.float32)
            v_t_target = _maybe_squeeze_last(v_t_target).astype(jnp.float32)

            next_state, next_ts = env.step(carry.state, action)

            # keep accounting consistent with A2C agent style (pmap-safe)
            new_acting_state = ActingState(
                state=next_state,
                timestep=next_ts,
                key=key,
                episode_count=carry.episode_count + jax.lax.psum(next_ts.last().sum(), "devices"),
                env_step_count=carry.env_step_count + jax.lax.psum(self.batch_size_per_device, "devices"),
            )

            reward = _maybe_squeeze_last(next_ts.reward).astype(jnp.float32)
            discount = _maybe_squeeze_last(next_ts.discount).astype(jnp.float32)

            transition = (obs, raw_action, logp, v_t_choice, v_t_target, reward, discount)
            return new_acting_state, transition

        keys = jax.random.split(acting_state.key, T)  # [T,2]
        new_acting_state, traj = jax.lax.scan(one_step, acting_state, keys)

        obs_T, raw_action_T, logp_T, v_t_choice,v_t_target, reward_T, discount_T = traj

        # Bootstrap at last observation after T steps
        last_obs = new_acting_state.timestep.observation
        last_value_choice, last_value_target = value_apply(params.critic, last_obs)
        last_value_choice = _maybe_squeeze_last(last_value_choice).astype(jnp.float32)
        last_value_target = _maybe_squeeze_last(last_value_target).astype(jnp.float32)
        rollout = RolloutTB(
            observation=obs_T,
            raw_action=raw_action_T,
            logp_old=logp_T.astype(jnp.float32),
            value_choice_old=v_t_choice.astype(jnp.float32),
            value_target_old=v_t_target.astype(jnp.float32),
            reward=reward_T.astype(jnp.float32),
            discount=discount_T.astype(jnp.float32),
        )
        return new_acting_state, rollout, last_value_choice,last_value_target

    # -----------------------------
    # Build flat batch for dhvl
    # -----------------------------
    def _build_batch(self, rollout: RolloutTB, last_value_choice: jnp.ndarray,
                     last_value_target: jnp.ndarray) -> dhvlBatchN:
        rewards = rollout.reward
        discounts = rollout.discount
        v_choice_old_TB = rollout.value_choice_old
        v_target_old_TB = rollout.value_target_old

        gamma = float(getattr(self, "discount_factor", 0.99))
        lam = float(self.dhvl_cfg.bootstrapping_factor)
        adv_TB, ret_choice_TB, ret_target_TB = _compute_highway_v_learning(
            rewards, discounts,
            v_choice_old_TB, v_target_old_TB,
            last_value_choice, last_value_target,
            gamma=gamma, lam=lam
        )

        if self.dhvl_cfg.normalize_adv:
            adv_flat = adv_TB.reshape((-1,))
            adv_TB = (adv_TB - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        T, B = int(rewards.shape[0]), int(rewards.shape[1])
        obs_N = _tree_reshape_TB_to_N(rollout.observation, T, B)
        raw_action_N = _tree_reshape_TB_to_N(rollout.raw_action, T, B)

        batch = dhvlBatchN(
            observation=obs_N,
            raw_action=raw_action_N,
            logp_old=jax.lax.stop_gradient(rollout.logp_old.reshape((T * B,))),
            value_choice_old=jax.lax.stop_gradient(v_choice_old_TB.reshape((T * B,))),
            value_target_old=jax.lax.stop_gradient(v_target_old_TB.reshape((T * B,))),
            advantage=jax.lax.stop_gradient(adv_TB.reshape((T * B,))),
            returns_choice=jax.lax.stop_gradient(ret_choice_TB.reshape((T * B,))),
            returns_target=jax.lax.stop_gradient(ret_target_TB.reshape((T * B,))),
        )
        return batch

    # -----------------------------
    # Loss on one minibatch
    # -----------------------------
    def _dhvl_loss(
        self,
        params,
        batch_mb: dhvlBatchN,            # each field has leading dim [M] where M=minibatch_size
        mb_mask: jnp.ndarray,           # [M] float32 {0,1}
        rng: jax.Array,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        dist = self.actor_critic_networks.parametric_action_distribution
        policy_apply = self.actor_critic_networks.policy_network.apply
        value_apply = self.actor_critic_networks.value_network.apply

        # Policy forward
        logits = policy_apply(params.actor, batch_mb.observation)
        mask = _extract_action_mask(batch_mb.observation)
        logits = _apply_action_mask_to_logits(logits, mask)

        logp_new = dist.log_prob(logits, batch_mb.raw_action).astype(jnp.float32)
        ratio = jnp.exp(logp_new - batch_mb.logp_old)

        eps = float(self.dhvl_cfg.clip_eps)
        adv = batch_mb.advantage

        pg_loss_unclipped = ratio * adv
        pg_loss_clipped = jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * adv
        pg_loss = -_masked_mean(jnp.minimum(pg_loss_unclipped, pg_loss_clipped), mb_mask)

        # Value forward
        v_choice_new, v_target_new = value_apply(params.critic, batch_mb.observation)
        v_choice_new = _maybe_squeeze_last(v_choice_new).astype(jnp.float32)
        v_target_new = _maybe_squeeze_last(v_target_new).astype(jnp.float32)

        if self.dhvl_cfg.clip_value_loss:
            v_eps = float(self.dhvl_cfg.value_clip_eps) if self.dhvl_cfg.value_clip_eps is not None else eps

            # --- choice head ---
            v_choice_old = batch_mb.value_choice_old.astype(jnp.float32)
            ret_choice = batch_mb.returns_choice.astype(jnp.float32)

            v_choice_clipped = v_choice_old + jnp.clip(v_choice_new - v_choice_old, -v_eps, v_eps)
            v_loss1_c = jnp.square(v_choice_new - ret_choice)
            v_loss2_c = jnp.square(v_choice_clipped - ret_choice)
            v_loss_choice = 0.5 * _masked_mean(jnp.maximum(v_loss1_c, v_loss2_c), mb_mask)

            # --- target head ---
            v_target_old = batch_mb.value_target_old.astype(jnp.float32)
            ret_target = batch_mb.returns_target.astype(jnp.float32)

            v_target_clipped = v_target_old + jnp.clip(v_target_new - v_target_old, -v_eps, v_eps)
            v_loss1_t = jnp.square(v_target_new - ret_target)
            v_loss2_t = jnp.square(v_target_clipped - ret_target)
            v_loss_target = 0.5 * _masked_mean(jnp.maximum(v_loss1_t, v_loss2_t), mb_mask)

        else:
            v_loss_choice = 0.5 * _masked_mean(jnp.square(v_choice_new - batch_mb.returns_choice), mb_mask)
            v_loss_target = 0.5 * _masked_mean(jnp.square(v_target_new - batch_mb.returns_target), mb_mask)


        v_loss = 0.5 * (v_loss_choice + v_loss_target)

        # Entropy bonus
        rng, ent_key = jax.random.split(rng)
        entropy = dist.entropy(logits, ent_key).astype(jnp.float32)
        ent = _masked_mean(entropy, mb_mask)

        total_loss = pg_loss + self.dhvl_cfg.vf_coef * v_loss - self.dhvl_cfg.ent_coef * ent

        approx_kl = _masked_mean(batch_mb.logp_old - logp_new, mb_mask)
        clipfrac = _masked_mean((jnp.abs(ratio - 1.0) > eps).astype(jnp.float32), mb_mask)

        metrics = {
            "loss/total": total_loss,
            "loss/policy": pg_loss,
            "loss/value": v_loss,
            "loss/value_choice": v_loss_choice,
            "loss/value_target": v_loss_target,
            "stats/entropy": ent,
            "stats/approx_kl": approx_kl,
            "stats/clipfrac": clipfrac,
            "stats/adv_mean": _masked_mean(adv, mb_mask),
            "stats/ratio_mean": _masked_mean(ratio, mb_mask),
            "stats/value_choice_mean": _masked_mean(v_choice_new, mb_mask),
            "stats/value_target_mean": _masked_mean(v_target_new, mb_mask),
            "stats/ret_choice_mean": _masked_mean(batch_mb.returns_choice, mb_mask),
            "stats/ret_target_mean": _masked_mean(batch_mb.returns_target, mb_mask),
        }
        return total_loss, metrics

    # -----------------------------
    # Main API: run_epoch (do not rename)
    # -----------------------------
    def run_epoch(self, training_state):
        params = training_state.params_state.params
        opt_state = training_state.params_state.opt_state
        update_count = training_state.params_state.update_count
        acting_state = training_state.acting_state

        acting_state, rollout, last_value_choice, last_value_target = self._rollout(params, acting_state)
        batchN = self._build_batch(rollout, last_value_choice, last_value_target)

        N = int(batchN.logp_old.shape[0])
        mb = int(self.dhvl_cfg.minibatch_size)
        if mb <= 0:
            mb = N
        num_minibatches = (N + mb - 1) // mb
        update_epochs = int(self.dhvl_cfg.update_epochs)

        global_step0 = jnp.asarray(update_count, dtype=jnp.int32)

        d = int(self.dhvl_cfg.policy_delay)
        d = 1 if d <= 0 else d  # 防止除0

        def zero_tree(t):
            return jax.tree_util.tree_map(jnp.zeros_like, t)

        def one_epoch(carry, key_epoch):
            params, opt_state, key, global_step = carry
            key, perm_key, mbkey = jax.random.split(key, 3)

            idx_mat, mask_mat = _make_minibatch_indices(perm_key, N, mb)  # [M,mb], [M,mb]
            mb_keys = jax.random.split(mbkey, num_minibatches)

            def one_minibatch(carry2, xs):
                params, opt_state, global_step = carry2
                idx, m_mask, k = xs

                # -------- helpers --------
                def zero_tree(t):
                    return jax.tree_util.tree_map(jnp.zeros_like, t)

                def _gather(tree):
                    def g(x):
                        # only gather arrays shaped [N, ...]
                        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == N:
                            return x[idx]
                        return x

                    return jax.tree_util.tree_map(g, tree)

                # -------- minibatch --------
                mbatch = dhvlBatchN(
                    observation=_gather(batchN.observation),
                    raw_action=_gather(batchN.raw_action),
                    logp_old=batchN.logp_old[idx],
                    value_choice_old=batchN.value_choice_old[idx],
                    value_target_old=batchN.value_target_old[idx],
                    returns_choice=batchN.returns_choice[idx],
                    returns_target=batchN.returns_target[idx],
                    advantage=batchN.advantage[idx],
                )

                def loss_fn(p):
                    return self._dhvl_loss(p, mbatch, m_mask, k)

                (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

                # -------- delayed actor update --------
                d_int = int(d)
                do_actor = (global_step % d_int) == 0
                do_actor = do_actor.astype(jnp.bool_)  # keep it as a scalar bool

                grads_actor = jax.lax.cond(
                    do_actor,
                    lambda g: g,
                    lambda g: zero_tree(g),
                    grads.actor,
                )
                grads = grads._replace(actor=grads_actor)

                # -------- optimizer step --------
                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                params = jax.tree_util.tree_map(lambda w, u: w + u, params, updates)

                # -------- metrics --------
                metrics = dict(metrics)
                metrics["stats/do_actor_update"] = do_actor.astype(jnp.float32)

                global_step = global_step + 1
                return (params, opt_state, global_step), metrics

            (params, opt_state, global_step), metrics_seq = jax.lax.scan(
                one_minibatch, (params, opt_state, global_step), (idx_mat, mask_mat, mb_keys)
            )
            metrics_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics_seq)
            metrics_mean["stats/do_actor_update_rate"] = jnp.mean(metrics_seq["stats/do_actor_update"])
            return (params, opt_state, key, global_step), metrics_mean

        key = acting_state.key
        key, epoch_key = jax.random.split(key)
        epoch_keys = jax.random.split(epoch_key, update_epochs)

        (params, opt_state, key, global_step), metrics_seq = jax.lax.scan(
            one_epoch, (params, opt_state, key, global_step0), epoch_keys
        )
        metrics = jax.tree_util.tree_map(lambda x: x[-1], metrics_seq)

        acting_state = acting_state._replace(key=key)

        new_params_state = training_state.params_state._replace(
            params=params,
            opt_state=opt_state,
            update_count=update_count + (update_epochs * num_minibatches),
        )
        new_state = training_state._replace(params_state=new_params_state, acting_state=acting_state)
        return new_state, metrics

