
from flax import nnx
import jax.numpy as jnp
import jax



class RMSNorm(nnx.Module): 
    def __init__(self, hidden_size: int, eps: float = 1e-9):
        super().__init__()
        self.weights = nnx.Param(jnp.ones((hidden_size, ), dtype=jnp.float32))
        self.eps = eps

    def __call__(self, hidden_states: jax.Array):
        dtype = hidden_states.dtype
        hidden_states_f32 = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.pow(hidden_states_f32, 2), axis=-1, keepdims=True)
        hidden_states = hidden_states_f32 * jax.lax.rsqrt(variance + self.eps)
        return (hidden_states * self.weights).astype(dtype=dtype)

class AdaLayerNormZero(nnx.Module):
    """
    Adaptive Layer Normalization Zero-Initialized.
    Used in Diffusion Transformers (DiT) to modulate blocks conditioning on a timestep.
    Projects a condition vector `c` into shift, scale, and gate parameters.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype=jnp.float32, param_dtype=jnp.float32, rngs: nnx.Rngs | None = None):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(jax.random.key(0))
            
        self.eps = eps
        self.dtype = dtype
        
        # Linear layer mapping hidden_size (from timestep embedder) -> 6 * hidden_size
        # (2 for self-attention scale/shift, 2 for MLP scale/shift, 2 for gate parameters)
        # We initialize it with zeros so it starts as an identity block
        self.linear = nnx.Linear(
            in_features=hidden_size, 
            out_features=6 * hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs 
        )
        
        # A standard layer norm applied to the input sequences before applying the adaptive scale/shift
        # LayerNorm differs from RMSNorm as it centers the mean
        self.norm = nnx.LayerNorm(num_features=hidden_size, epsilon=eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x: jax.Array, c: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Args:
            x: Input sequence tensor (B, S, D)
            c: Conditioning tensor (e.g., Timestep Embedding) (B, D)
            
        Returns:
            x_norm: Unmodulated layer norm output to be scaled/shifted manually by the block.
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        """
        # 1. Project condition -> 6 modulating parameters (B, 6*D)
        emb_out = self.linear(jax.nn.silu(c))
        
        # 2. Reshape and split into the 6 components
        # emb_out shape: (B, 6, D) or we can just unpack across axis -1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(emb_out, 6, axis=-1)
        
        scale_msa = scale_msa[:, None, :] # Broadcast to (B, 1, D) for sequence length S
        shift_msa = shift_msa[:, None, :]
        gate_msa = gate_msa[:, None, :]
        
        scale_mlp = scale_mlp[:, None, :]
        shift_mlp = shift_mlp[:, None, :]
        gate_mlp = gate_mlp[:, None, :]
        
        # Return the normalized x and the components so the DiTBlock can apply them
        return self.norm(x), shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp