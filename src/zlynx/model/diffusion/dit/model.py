import jax
import jax.numpy as jnp
from flax import nnx

from ...modules.attn import Attention
from ...modules.mlp import MLP
from ...modules.norm import AdaLayerNormZero
from ...modules.embed import PatchEmbed, TimestepEmbedder
from .config import DiTConfig

class DiTBlock(nnx.Module):
    """
    A DiT block with adaptive layer norm zero (AdaLN-Zero) conditioning.
    """
    def __init__(self, config: DiTConfig, dtype=jnp.float32, param_dtype=jnp.float32, rngs: nnx.Rngs | None = None):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(jax.random.key(0))
            
        self.norm1 = AdaLayerNormZero(config.hidden_size, eps=1e-6, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        
        self.attn = Attention(
            hidden_size=config.hidden_size,
            attention_head=config.num_heads,
            head_dim=config.hidden_size // config.num_heads,
            bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            key=rngs.params()
        )
        
        self.norm2 = nnx.LayerNorm(num_features=config.hidden_size, epsilon=1e-6, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_dize=int(config.hidden_size * config.mlp_ratio), # Preserved variable typo from module constructor
            act_fn=nnx.gelu,
            bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            key=rngs.params()
        )

    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        # AdaLN-Zero modulates both Attention and MLP through a single projection inside norm1
        norm_x, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(x, c)
        
        # Self-attention branch (incorporate scaling and shifting directly in skip-connection space)
        attn_out, _ = self.attn(norm_x * (1 + scale_msa) + shift_msa)
        x = x + gate_msa * attn_out
        
        # MLP branch
        norm_x2 = self.norm2(x)
        x = x + gate_mlp * self.mlp(norm_x2 * (1 + scale_mlp) + shift_mlp)
        
        return x

class FinalLayer(nnx.Module):
    """
    The final layer of DiT.
    Applies a standard LayerNorm, shifts/scales from the timestep conditioning, 
    and maps the hidden sequence dimensions back to spatial patch pixels.
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, dtype=jnp.float32, param_dtype=jnp.float32, rngs: nnx.Rngs | None = None):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(jax.random.key(0))
            
        self.norm_final = nnx.LayerNorm(num_features=hidden_size, epsilon=1e-6, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        
        # Computes shift and scale parameters purely for the final output space mapping
        self.linear = nnx.Linear(
            in_features=hidden_size,
            out_features=2 * hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs
        )
        
        # The ultimate reverse-patchifier tensor transformation
        self.linear_out = nnx.Linear(
            in_features=hidden_size,
            out_features=patch_size * patch_size * out_channels,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs
        )

    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        # Project conditioning vector to a shift & scale (B, 2*D) -> split
        emb_out = self.linear(jax.nn.silu(c))
        shift, scale = jnp.split(emb_out, 2, axis=-1)
        
        # Broadcast across sequence length
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        
        x = self.norm_final(x)
        x = x * (1 + scale) + shift
        x = self.linear_out(x)
        return x

class DiT(nnx.Module):
    """
    Diffusion Transformer (DiT).
    Takes noisy continuous paths and timesteps to project un-noising predictions.
    """
    def __init__(self, config: DiTConfig, dtype=jnp.float32, param_dtype=jnp.float32, rngs: nnx.Rngs | None = None):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(jax.random.key(0))
            
        self.config = config
        
        self.x_embedder = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
        
        self.t_embedder = TimestepEmbedder(
            hidden_size=config.hidden_size,
            frequency_embedding_size=config.frequency_embedding_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
        
        # Fixed 2D Sine/Cosine Position Embeddings
        self.pos_embed = nnx.Param(
            jnp.zeros((1, self.x_embedder.num_patches, config.hidden_size), dtype=param_dtype) # Simplified learnable placeholder for now
        )
        
        # DiT backbone
        self.blocks = nnx.List([
            DiTBlock(config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            for _ in range(config.depth)
        ])
        
        self.final_layer = FinalLayer(
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            out_channels=config.in_channels,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """
        Args:
            x: (B, H, W, C) input spatial imagery with noise
            t: (B,) timesteps
        Returns:
            (B, N, patch_size * patch_size * C) flattened spatial sequences of predicted noise
        """
        # Embed physical image layout patches to unconstrained sequential tokens
        x = self.x_embedder(x) + self.pos_embed[...]
        
        # Project continuous timestep scalars to model `hidden_size`
        c = self.t_embedder(t)
        
        # Progress dynamically down the Transformer utilizing AdaLN
        for block in self.blocks:
            x = block(x, c)
            
        # Un-project sequence dimensions back to physical patch constraints via FinalLayer Modulations
        x = self.final_layer(x, c)
        return x
