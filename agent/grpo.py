# ppo_by_steps.py
import sys
from typing import Dict, Tuple, Any

import jax
from jax import lax
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
    percentile_p: int = 2
    mean_reward_method: bool = False
    reward_mode: str = "dense"

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

# ===================================================================
# GRPO Agent 实现
# ===================================================================

class GRPOAgent(A2CAgent):

    def __init__(self, grpo_cfg: GRPOConfig, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.grpo_cfg = grpo_cfg
        self.batch_size = int(getattr(self, "total_batch_size", kwargs.get("total_batch_size")))
    def _compute_advantage_sparse_reward(self, params, data: Transition):
        r = data.reward.astype(jnp.float32) # (T, B)
        done = data.extras["done"].astype(bool) # (T, B)
        T, B = r.shape
        prev_done = jnp.roll(done, shift=1, axis=0).at[0, :].set(False)
        ep_ids = jnp.cumsum(prev_done.astype(jnp.int32), axis=0)  # (T, B)
        num_ep = jnp.sum(done, axis=0).astype(jnp.int32)  # (B,)  episode_id
        valid = (ep_ids < num_ep[None, :])  # (T, B)
        rb = jnp.swapaxes(r, 0, 1)  # (B, T)
        idsb = jnp.swapaxes(ep_ids, 0, 1)  # (B, T)
        vb = jnp.swapaxes(valid, 0, 1)  # (B, T)

        def per_batch_last(r_b, id_b, v_b, n_b, d_b):
            data_to_sum = (r_b * d_b.astype(jnp.float32) * v_b.astype(jnp.float32))
            ids = jnp.where(v_b, id_b, 0)
            all_ids = jnp.arange(T, dtype=jnp.int32)

            def _get_sum_for_one_id(i):
                return jnp.sum(data_to_sum * (ids == i))

            seg_last = jax.vmap(_get_sum_for_one_id)(all_ids)
            seg_last = jnp.where(all_ids < n_b, seg_last, 0.0)
            R_last = seg_last[ids] * v_b.astype(jnp.float32)
            return R_last, seg_last
        # seg shape: (B, T)  R_step (B, T)
        R_step_t, seg = jax.vmap(per_batch_last, in_axes=(0, 0, 0, 0, 0))(rb, idsb, vb, num_ep, jnp.swapaxes(done, 0, 1))
        if self.grpo_cfg.percentile_p > 0:
            p = self.grpo_cfg.percentile_p
            mask_e_episode = (jnp.arange(T)[None, :] < num_ep[:, None])  # (B, T)
            seg_masked = jnp.where(mask_e_episode, seg, jnp.inf)  # (B, T)
            sorted_seg = jnp.sort(seg_masked, axis=1)  # (B, T)
            # num_ep: (B,)
            num_ep_f = num_ep.astype(jnp.float32)
            # 找到percentile的值
            idx_f = p * jnp.maximum(num_ep_f - 1.0, 0.0)
            idx = jnp.floor(idx_f).astype(jnp.int32)  # (B,)
            idx = jnp.clip(idx, 0, T - 1)
            percentile = jnp.take_along_axis(
                sorted_seg,
                idx[:, None],  # (B, 1)
                axis=1,
            )  # (B, 1)
        mask_e = (jnp.arange(T)[None, :] < num_ep[:, None]).astype(jnp.float32)  # (B, T)
        count = jnp.maximum(num_ep.astype(jnp.float32), 1.0)[:, None] # B, 1
        mean = (seg * mask_e).sum(axis=1, keepdims=True) / count # B, 1
        mask = vb.astype(jnp.float32)

        if self.grpo_cfg.mean_reward_method:
            var = (((seg - mean) ** 2) * mask_e).sum(axis=1, keepdims=True) / count
            std = jnp.sqrt(var + 1e-8)
            advantage = ((R_step_t - mean) / std) * mask
        elif self.grpo_cfg.percentile_p > 0:
            var = (((seg - percentile) ** 2) * mask_e).sum(axis=1, keepdims=True) / count
            std = jnp.sqrt(var + 1e-8)
            advantage = ((R_step_t - percentile) / std) * mask
        advantage = jnp.swapaxes(advantage, 0, 1)  # (T, B)

        targets = jnp.zeros_like(r)
        return advantage, targets, num_ep.mean()

    def _compute_advantage_dense_reward(self, params, data: Transition):
        r = data.reward.astype(jnp.float32)  # (T, B)
        done = data.extras["done"].astype(bool)  # (T, B)
        T, B = r.shape

        # metric: 每个并行环境在 rollout 内完整结束的 episode 数
        num_ep = jnp.sum(done, axis=0).astype(jnp.int32)  # (B,)

        rb = jnp.swapaxes(r, 0, 1)  # (B, T)
        done_bt = jnp.swapaxes(done, 0, 1)  # (B, T)

        # ------------------------------------------------------------
        # 1) return-to-go: G_{i,t} = sum_{k=t}^T r_{i,k} (episode 内，遇到 done 截断)
        # ------------------------------------------------------------
        def per_batch_expected_reward(r_b, d_b):
            def _cum_step_reward(carry, inputs):
                r_t, done_t = inputs
                not_done = 1.0 - done_t.astype(jnp.float32)
                g_t = r_t + not_done * carry
                return g_t, g_t

            _, expected_reward = lax.scan(
                _cum_step_reward,
                0.0,
                (r_b, d_b),
                reverse=True
            )
            return expected_reward  # (T,)

        exp_bt = jax.vmap(per_batch_expected_reward, in_axes=(0, 0))(rb, done_bt)  # (B, T)

        # ------------------------------------------------------------
        # 2) 去掉最后一个不完整 episode：只保留 ep_ids < num_complete_eps
        # ------------------------------------------------------------
        prev_done = jnp.roll(done, shift=1, axis=0).at[0, :].set(False)  # (T,B)
        ep_ids = jnp.cumsum(prev_done.astype(jnp.int32), axis=0)  # (T,B)
        ep_ids_bt = jnp.swapaxes(ep_ids, 0, 1)  # (B,T)

        # 每个并行环境 b：完整 episode 数 = done 的次数
        n_ep_b = jnp.sum(done_bt, axis=1).astype(jnp.int32)  # (B,)
        valid_bt = (ep_ids_bt < n_ep_b[:, None])  # (B,T) True=属于完整episode

        # ------------------------------------------------------------
        # 3) 计算 episode 内 step index: pos（用于“同一 t 比较”）
        # ------------------------------------------------------------
        start = prev_done.at[0, :].set(True)  # (T,B) t=0 视为 episode 起点

        def step_pos(carry, start_t):
            pos_t = jnp.where(start_t, 0, carry + 1)
            return pos_t, pos_t

        _, pos_tb = lax.scan(step_pos, jnp.zeros((B,), jnp.int32), start)  # (T,B)
        pos_bt = jnp.swapaxes(pos_tb, 0, 1)  # (B,T)

        # ------------------------------------------------------------
        # 4) z_{i,t} = phi(s_{i,t})：用 data.observation 当 embedding（更贴近论文语义）
        # ------------------------------------------------------------
        mask_bt = jnp.swapaxes(data.observation.action_mask, 0, 1).astype(jnp.float32)  # (B,T,num_cities)
        pos_bt_int = jnp.swapaxes(data.observation.position, 0, 1).astype(jnp.int32)  # (B,T)
        num_cities = mask_bt.shape[-1]
        pos_oh_bt = jax.nn.one_hot(pos_bt_int, num_cities).astype(jnp.float32)  # (B,T,num_cities)
        # 拼成 embedding： (B,T,2*num_cities)
        z_bt = jnp.concatenate([mask_bt, pos_oh_bt], axis=-1)

        # ------------------------------------------------------------
        # 5) Kernel regression baseline（leave-one-out）+ 不除 std：
        #    mu_{i,t} = sum_{j!=i} w_ij,t * G_{j,t}
        #    adv = G_{i,t} - mu_{i,t}
        # ------------------------------------------------------------
        eps = 1e-8
        h_mult = float(getattr(self.grpo_cfg, "kernel_h_mult", 1.0))
        h_min = float(getattr(self.grpo_cfg, "kernel_h_min", 1e-6))

        def per_batch_kernel_adv(G_b, z_b, pos_b, epid_b, v_b):
            # G_b: (T,), z_b: (T,D), pos_b: (T,), epid_b: (T,), v_b: (T,) bool

            T_, D = z_b.shape
            # h = jnp.asarray(getattr(self.grpo_cfg, "kernel_h", 0.5 * jnp.sqrt(D)), dtype=jnp.float32)
            h = jnp.asarray(0.8, dtype=jnp.float32)
            h = jnp.maximum(h, h_min)

            diff = z_b[:, None, :] - z_b[None, :, :]
            dist2 = jnp.sum(diff * diff, axis=-1)
            K = jnp.exp(-dist2 / (2.0 * h * h))

            same_pos = (pos_b[:, None] == pos_b[None, :])  # 同一步（episode 内 step index）
            diff_ep = (epid_b[:, None] != epid_b[None, :])  # 不同 trajectory（不同 episode）
            vv = (v_b[:, None] & v_b[None, :])  # valid-valid
            loo = ~jnp.eye(T, dtype=bool)

            W = K * same_pos.astype(K.dtype) * diff_ep.astype(K.dtype) * vv.astype(K.dtype) * loo.astype(K.dtype)
            denom = jnp.sum(W, axis=1, keepdims=True)
            Wn = W / (denom + eps)

            mu = jnp.sum(Wn * G_b[None, :], axis=1)
            adv = G_b - mu
            adv = jnp.where((denom.squeeze(-1) > 0) & v_b, adv, 0.0)

            # -----------------------------
            # metrics: 权重分布熵（按行：每个 i 的权重分布）
            # -----------------------------
            w = Wn
            ent = -jnp.sum(w * jnp.log(w + 1e-12), axis=1)  # (T,)
            ent_valid = (denom.squeeze(-1) > 0) & v_b
            ent_sum = jnp.sum(jnp.where(ent_valid, ent, 0.0))
            ent_cnt = jnp.sum(ent_valid.astype(jnp.float32))

            # 也可以顺便统计“有效邻居数”（你原来就有；这里不改动逻辑）
            nnz = jnp.sum((W > 0).astype(jnp.float32), axis=1)  # (T,)
            nnz = jnp.where((denom.squeeze(-1) > 0) & v_b, nnz, 0.0)

            return adv, ent_sum, ent_cnt

        adv_bt, ent_sum_b, ent_cnt_b = jax.vmap(per_batch_kernel_adv, in_axes=(0, 0, 0, 0, 0))(
            exp_bt, z_bt, pos_bt, ep_ids_bt, valid_bt
        )

        advantage = jnp.swapaxes(adv_bt, 0, 1)  # (T,B)

        # 这个是最终要进 TensorBoard 的标量
        kernel_weight_entropy_mean = jnp.sum(ent_sum_b) / (jnp.sum(ent_cnt_b) + eps)

        targets = jnp.zeros_like(r)
        return advantage, targets, num_ep.mean(), kernel_weight_entropy_mean

    def _loss_from_data(self, params, acting_state: ActingState, data: Transition):
        policy_apply = self.actor_critic_networks.policy_network.apply

        def fwd(_, ob):
            return None, policy_apply(params.actor, ob)

        _, new_logits = jax.lax.scan(fwd, None, data.observation)
        mask_b = data.extras.get("action_mask", _tree_true_like(new_logits))
        masked_new_logits = _apply_mask_to_logits(new_logits, mask_b)
        dist = self.actor_critic_networks.parametric_action_distribution
        logp_new = dist.log_prob(masked_new_logits, data.extras.get("raw_action"))
        ratio = jnp.exp(logp_new - data.log_prob)
        if self.grpo_cfg.reward_mode == "dense":
            advantage, targets, num_ep_mean, kernel_weight_entropy_mean = self._compute_advantage_dense_reward(params, data)
        elif self.grpo_cfg.reward_mode == "sparse":
            # advantage, targets, num_ep_mean = self._compute_advantage_sparse_reward(params, data)  # T, B
            advantage, targets, num_ep_mean, kernel_weight_entropy_mean = self._compute_advantage_dense_reward(params,
                                                                                                               data)

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
            num_ep_mean=num_ep_mean,
            **{"kernel/weight_entropy": kernel_weight_entropy_mean},
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
            raw_action_t = jax.vmap(dist.sample_no_postprocessing, in_axes=(0, 0))(masked_logits_t, actor_keys)
            logp_t = dist.log_prob(masked_logits_t, raw_action_t)
            action_t = dist.postprocess(raw_action_t)
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
            obs_t, action_t, raw_action_t,timestep.reward, timestep.discount, next_obs_t, logp_t, mask_t, logits_t, newly_done)
            carry_out = (key, next_state, next_obs_t)
            return carry_out, transition

        init_carry = (key, state, first_obs)
        (_, _, _), traj = jax.lax.scan(step_fn, init_carry, None, length=T_max)
        obs_b, action_b, raw_action_b, reward_b, discount_b, next_obs_b, logp_b, mask_b, logits_t, dones_b = traj

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
                "raw_action": raw_action_b,
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

            #  L2 norm
            actor_grads_leaves = jax.tree_util.tree_leaves(grads.actor)
            actor_global_norm = jnp.sqrt(
                sum(jnp.sum(jnp.square(x)) for x in actor_grads_leaves if x is not None)
            )
            inner_metrics['grad_norm/actor_global_norm'] = actor_global_norm
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = jax.tree.map(lambda w, u: w + u, params, updates)
            metrics = inner_metrics
        new_params_state = training_state.params_state._replace(
            params=params, opt_state=opt_state, update_count=update_count + K
        )

        new_state = training_state._replace(params_state=new_params_state, acting_state=acting_state)
        return new_state, metrics
