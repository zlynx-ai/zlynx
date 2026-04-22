from dataclasses import dataclass, field
from flax import struct
from typing import Literal


AttentionType = Literal["local", "global"]


@struct.dataclass
class Gemma4Config:
    arch: str = "Gemma4LanguageModel"
    conf: str = "Gemma4Config"

    # Embedding
    vocab_size: int = 262144
    embed_dim: int = 2048

    # Attention
    num_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 256
    num_global_kv_heads: int | None = None
    global_key_size: int = 512
    k_eq_v_global: bool = False

    # FFW
    hidden_dim: int = 8192

    # Layers
    num_layers: int = 18
    attention_pattern: tuple[AttentionType, ...] = ("local", "local", "local", "local", "global")

    # RoPE
    local_base_frequency: int = 10_000
    global_base_frequency: int = 1_000_000
    local_rope_proportion: float = 1.0
    global_rope_proportion: float = 0.25

    # Norms
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True
    qk_norm: bool = True

    # Logits
    final_logit_softcap: float | None = 30.0
    attn_logits_softcap: float | None = None

    # Sliding window (local attention)
    sliding_window_size: int = 1024

    # MoE
    enable_moe: bool = False
    num_experts: int = 0
    expert_dim: int = 0
    top_k_experts: int = 8
    moe_dense_hidden_dim: int = 0

    # dtype
    dtype: str = "bfloat16"
    param_dtype: str = "bfloat16"

    def attention_types(self) -> tuple[AttentionType, ...]:
        """Expand attention_pattern to full num_layers sequence."""
        pattern = self.attention_pattern
        n = len(pattern)
        full = pattern * (self.num_layers // n)
        if self.num_layers % n:
            full += pattern[: self.num_layers % n]
        return full


# Preset configs matching gemma4 model sizes

Gemma4_E2BConfig = Gemma4Config(
    embed_dim=1536,
    num_heads=8,
    num_kv_heads=1,
    hidden_dim=1536 * 4,
    num_layers=35,
    attention_pattern=("local", "local", "local", "local", "global"),
    sliding_window_size=512,
    global_key_size=512,
    k_eq_v_global=False,
)

Gemma4_E4BConfig = Gemma4Config(
    embed_dim=2560,
    num_heads=8,
    num_kv_heads=2,
    hidden_dim=2560 * 4,
    num_layers=42,
    attention_pattern=("local", "local", "local", "local", "local", "global"),
    sliding_window_size=512,
    global_key_size=512,
    k_eq_v_global=False,
)

Gemma4_31BConfig = Gemma4Config(
    embed_dim=5376,
    num_heads=32,
    num_kv_heads=16,
    num_global_kv_heads=4,
    hidden_dim=5376 * 4,
    num_layers=60,
    attention_pattern=("local", "local", "local", "local", "local", "global"),
    sliding_window_size=1024,
    global_key_size=512,
    k_eq_v_global=True,
)

Gemma4_26B_A4BConfig = Gemma4Config(
    embed_dim=2816,
    num_heads=16,
    num_kv_heads=8,
    num_global_kv_heads=2,
    hidden_dim=2112,
    num_layers=30,
    attention_pattern=("local", "local", "local", "local", "local", "global"),
    sliding_window_size=1024,
    global_key_size=512,
    k_eq_v_global=True,
    enable_moe=True,
    num_experts=128,
    expert_dim=704,
    top_k_experts=8,
    moe_dense_hidden_dim=2112,
)
