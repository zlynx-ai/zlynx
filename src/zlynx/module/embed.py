import jax
import jax.numpy as jnp
from flax import nnx

class TimestepEmbedder(nnx.Module):
    """
    Embeds scalar timesteps into a dense vector space using Sinusoidal encodings followed by an MLP.
    Standard in Diffusion Models (DiT).
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, dtype=jnp.float32, param_dtype=jnp.float32, rngs: nnx.Rngs | None = None):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(jax.random.key(0))
        
        self.frequency_embedding_size = frequency_embedding_size
        
        self.mlp = nnx.Sequential(
            nnx.Linear(frequency_embedding_size, hidden_size, use_bias=True, dtype=dtype, param_dtype=param_dtype, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size, use_bias=True, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        )

    def __call__(self, t: jax.Array) -> jax.Array:
        """
        Args:
            t: (Batch_Size, ) tensor of continuous or discrete timestep values.
        Returns:
            (Batch_Size, hidden_size) conditioning vector.
        """
        half_dim = self.frequency_embedding_size // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        
        # Pad if the desired dimension is odd
        if self.frequency_embedding_size % 2 == 1:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
            
        return self.mlp(emb)


class PatchEmbed(nnx.Module):
    """
    2D Image to Patch Embedding.
    Equivalent to the Vision Transformer (ViT) stem.
    Slices an image into non-overlapping patches and linearly projects each to the hidden size.
    """
    def __init__(
        self, 
        img_size: int = 256, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        embed_dim: int = 768, 
        dtype=jnp.float32, 
        param_dtype=jnp.float32, 
        rngs: nnx.Rngs | None = None
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(jax.random.key(0))
            
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (B, H, W, C) Image tensor.
        Returns:
            (B, N, D) Sequence of tokens.
        """
        B, H, W, C = x.shape
        x = self.proj(x) # (B, H/P, W/P, D)
        # Flatten spatial dimensions into a single sequence length (N = (H/P) * (W/P))
        x = x.reshape(B, self.num_patches, -1) 
        return x
