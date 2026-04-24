import jax
import jax.numpy as jnp
import time
from zlynx.model.meta.llama import LlamaConfig, LlamaLanguageModel

def test_fast_generation():
    print("\n=== Testing Generation with External KV Cache ===")

    # Tiny model for quick testing
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        head_dim=16
    )

    # Initialize Model
    key = jax.random.key(42)
    model = LlamaLanguageModel(config, key=key)

    # Create a batch of dummy prompts
    batch_size = 2
    prompt_len = 16
    max_new_tokens = 32
    ctxlen = 64

    prompts = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)

    print("\n1. Warming up (prefill + first decode compilation)...")
    start_compile = time.time()
    out = model.generate(prompts, max_new_tokens=max_new_tokens, ctxlen=ctxlen)
    jax.block_until_ready(out)
    print(f"First run finished in {time.time() - start_compile:.2f} seconds.")

    print("\n2. Benchmarking (cached JIT)...")
    start_run = time.time()
    out = model.generate(prompts, max_new_tokens=max_new_tokens, ctxlen=ctxlen)
    jax.block_until_ready(out)
    elapsed = time.time() - start_run

    total_generated_tokens = batch_size * max_new_tokens
    tps = total_generated_tokens / elapsed

    print(f"Generated {total_generated_tokens} tokens in {elapsed:.4f} seconds.")
    print(f"Throughput: {tps:.2f} tokens/second.")

    assert out.shape == (batch_size, ctxlen), f"Expected ({batch_size}, {ctxlen}), got {out.shape}"
    # Verify prompt tokens preserved
    assert jnp.all(out[:, :prompt_len] == prompts), "Prompt tokens should be preserved"
    print("\nGeneration verified successfully!")

if __name__ == "__main__":
    test_fast_generation()
