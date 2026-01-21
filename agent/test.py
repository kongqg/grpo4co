@jax.jit
def total_loss(self, traj, grad_params):
    n = len(traj)

    loss = 0.0

    for k in range(n - 1, -1, -1):

        s_k = traj[k]["observation"]
        a_k = traj[k]["action"]
        r_k = traj[k]["reward"]
        next_s_k = traj[k]["next_observation"]

        next_v = self.network.select("value")(next_s_k)
        if k == n - 1:
            # Gn-1 = rn-1 + γ * V
            G_k = r_k + self.config['discount'] * next_v
            G = G_k
        else:
            # Gk-1 = V_θ + λ * adjustment
            u = G - next_v  # G = Gk (the prev step)
            G_k_1 = next_v + self.config['elambda'] * (
                        self.config['expectile'] * jnp.maximum(u, 0) - (1 - self.config['expectile']) * jnp.maximum(-u,
                                                                                                                    0))
            G = G_k_1

        # value_loss
        traget_qs = self.network.select("target_critic")(s_k, actions=a_k)
        target_q = jnp.min(traget_qs, axis=0, keepdims=False)
        v = self.network.select("value")(s_k, params=grad_params)
        lam = 1 / jax.lax.stop_gradient(jnp.abs(v).mean())
        value_loss = self.expectile_loss(target_q - v, target_q - v, self.config['expectile']).mean()

        if self.config['normalize_q_loss']:
            value_loss = lam * value_loss

        # critic_loss
        qs = self.network.select('critic')(s_k, actions=a_k, params=grad_params)
        critic_loss = ((qs - G) ** 2).mean()

        loss = loss + value_loss + critic_loss
    return loss