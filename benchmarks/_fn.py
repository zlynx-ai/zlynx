import time
import jax
import numpy as np
from flax.struct import dataclass

def bench(fn, *args, warmup=5, repeat=50):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": np.mean(times).item() * 1000,
        "median_ms": np.median(times).item() * 1000,
        "min_ms": np.min(times).item() * 1000,
        "max_ms": np.max(times).item() * 1000,
    }

def random_input(len, size, dtype=None):

    if dtype is None:
        dtype = "float32"

    k = jax.random.key(0)
    return jax.random.uniform(k, (len, size), dtype=dtype)

@dataclass
class TestConfig:
    vocab_size: int = 80000
    hidden_size: int = 1024
    intermediate_size: int = 2048
    act_fn: str = "silu"
    dtype: str = "float32"
    param_dtype: str = "float32"
    # attention_head: int = 16
    # kv_head: int = 8
    # head_dim: int = 32

    # rope
    rope_theta: float = 10_000
    original_max_position_embedding: int = 2048
    max_position_embedding: int = 2048
    rope_scaling: dict | None = None