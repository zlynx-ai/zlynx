from dataclasses import dataclass

from flax import struct

@struct.dataclass
class DiTConfig:
    """Config for Diffusion Transformer (DiT)."""
    img_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    frequency_embedding_size: int = 256
    use_bias: bool = True
