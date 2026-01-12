import jax.numpy as jnp
from jax import jit, lax

# 假设您的 ep_ids 是从 (T, B) 展平来的
# ep_ids 示例 (已扁平化): [0, 0, 0, 1, 1, 1, 1, 2, 2]
# T=3, B=3
T, B = 3, 3
prev_done = jnp.array([
    [0, 0, 0],
    [1, 1, 0],
    [0, 0, 1]
])  # 假设 [T, B]
ep_ids_2d = jnp.cumsum(prev_done.astype(jnp.int32), axis=0)  # (T, B)
ep_ids_flat = ep_ids_2d.flatten()  # (T * B,)
print(ep_ids_flat)

result = jnp.bincount(ep_ids_flat)
result2 = jnp.max(result)
print(result2)