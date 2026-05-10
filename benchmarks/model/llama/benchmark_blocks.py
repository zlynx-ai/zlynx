from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.model.llama._transformer import LlamaBlock, LlamaBlock_N
from zlynx.module import RotaryEmbedding


@dataclasses.dataclass(frozen=True)
class BlockBenchConfig:
    hidden_size: int = 512
    intermediate_size: int = 2048
    attention_head: int = 8
    kv_head: int = 8
    head_dim: int = 64
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    max_position_embedding: int = 2048
    original_max_position_embedding: int = 2048
    rope_scaling: dict | None = None
    act_fn: str = "silu"
    bias: bool = False
    dtype: str = "bfloat16"
    param_dtype: str = "float32"


def parse_dtype(value: str) -> jnp.dtype:
    if value == "float32":
        return jnp.float32
    if value == "bfloat16":
        return jnp.bfloat16
    if value == "float16":
        return jnp.float16
    raise argparse.ArgumentTypeError(
        "dtype must be one of: float32, bfloat16, float16"
    )


def make_config(args: argparse.Namespace) -> BlockBenchConfig:
    if args.hidden_size % args.attention_head != 0:
        raise ValueError("hidden_size must be divisible by attention_head.")

    head_dim = args.head_dim or args.hidden_size // args.attention_head
    return BlockBenchConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        attention_head=args.attention_head,
        kv_head=args.kv_head or args.attention_head,
        head_dim=head_dim,
        dtype=args.dtype,
        param_dtype=args.param_dtype,
    )


def make_inputs(
    config: BlockBenchConfig,
    *,
    batch_size: int,
    seq_len: int,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, jax.Array]]:
    key = jax.random.key(0)
    hidden_states = jax.random.normal(
        key, (batch_size, seq_len, config.hidden_size), dtype=dtype
    )

    attention_mask = nnx.combine_masks(
        nnx.make_attention_mask(
            jnp.ones((batch_size, seq_len), dtype=jnp.bool_),
            jnp.ones((batch_size, seq_len), dtype=jnp.bool_),
        ),
        nnx.make_causal_mask(jnp.ones((batch_size, seq_len), dtype=jnp.bool_)),
    )

    position_ids = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
    rotary = RotaryEmbedding(
        config.rope_theta,
        config.head_dim,
        config.max_position_embedding,
        config.rope_scaling,
    )
    position_embedding = rotary(hidden_states, position_ids)
    return hidden_states, attention_mask, position_embedding


def block_until_ready(value):
    return jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def benchmark(
    name: str,
    fn: Callable[[], jax.Array],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, float | str]:
    for _ in range(warmup):
        block_until_ready(fn())

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1000)

    return {
        "name": name,
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def format_result(result: dict[str, float | str], baseline_ms: float | None = None) -> str:
    speedup = ""
    if baseline_ms is not None:
        speedup = f" | speedup={baseline_ms / result['mean_ms']:.2f}x"

    return (
        f"{result['name']:<14} "
        f"mean={result['mean_ms']:.3f} ms | "
        f"median={result['median_ms']:.3f} ms | "
        f"min={result['min_ms']:.3f} ms | "
        f"max={result['max_ms']:.3f} ms"
        f"{speedup}"
    )


def run(args: argparse.Namespace) -> None:
    dtype = parse_dtype(args.dtype)
    config = make_config(args)
    hidden_states, attention_mask, position_embedding = make_inputs(
        config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=dtype,
    )

    pallas_block = LlamaBlock(config, rngs=nnx.Rngs(args.seed), layer_idx=0)
    normal_block = LlamaBlock_N(config, rngs=nnx.Rngs(args.seed), layer_idx=0)

    @nnx.jit
    def run_pallas(block, x, mask, pos):
        return block(x, mask, pos)[0]

    @nnx.jit
    def run_normal(block, x, mask, pos):
        return block(x, mask, pos)[0]

    def pallas_fn():
        return run_pallas(pallas_block, hidden_states, attention_mask, position_embedding)

    def normal_fn():
        return run_normal(normal_block, hidden_states, attention_mask, position_embedding)

    print("=== LlamaBlock benchmark ===")
    print(f"device       : {jax.devices()[0].platform} / {jax.devices()[0]}")
    print(f"batch x seq  : {args.batch_size} x {args.seq_len}")
    print(f"hidden       : {config.hidden_size}")
    print(f"intermediate : {config.intermediate_size}")
    print(f"heads        : {config.attention_head} q / {config.kv_head} kv")
    print(f"dtype        : {args.dtype}")
    print(f"warmup/repeat: {args.warmup}/{args.repeat}")
    print()

    results = []
    for name, fn in (("pallas", pallas_fn), ("normal", normal_fn)):
        try:
            result = benchmark(name, fn, warmup=args.warmup, repeat=args.repeat)
        except Exception as exc:
            print(f"{name:<14} failed: {type(exc).__name__}: {exc}")
            continue
        results.append(result)

    if not results:
        raise SystemExit(1)

    baseline_ms = next(
        (result["mean_ms"] for result in results if result["name"] == "normal"),
        None,
    )
    for result in results:
        print(format_result(result, baseline_ms if result["name"] != "normal" else None))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare benchmark LlamaBlock with Pallas vs LlamaBlock_N."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--attention-head", type=int, default=8)
    parser.add_argument("--kv-head", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--param-dtype", default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
