




from typing import Sequence, Any
import jax
import jax.numpy as jnp
from flax import nnx

class LoraLinear(nnx.Module):
    """Low-Rank Adaptation wrapper for nnx.Linear"""
    def __init__(self, base_layer: nnx.Linear, r: int, alpha: int, rngs: nnx.Rngs):
        super().__init__()
        # freeze the base layer parameters by casting them to fixed variables
        # Note: if the base layer's kernel/bias are nnx.Param, changing them to nnx.Variable removes them from optimizer state!
        self.base_kernel = nnx.Variable(base_layer.kernel.get_value())
        if base_layer.bias is not None:
            self.base_bias = nnx.Variable(base_layer.bias.get_value())
        else:
            self.base_bias = None

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = self.base_kernel.get_value().shape[0]
        out_features = self.base_kernel.get_value().shape[1]
        
        # A matrix: (in_features, r) - initialized with normal
        self.lora_A = nnx.Param(jax.random.normal(rngs.params(), (in_features, r), dtype=base_layer.param_dtype))
        
        # B matrix: (r, out_features) - initialized to zero
        self.lora_B = nnx.Param(jnp.zeros((r, out_features), dtype=base_layer.param_dtype))
        
        self.dtype = base_layer.dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        # standard forward pass
        base_out = jnp.dot(x, self.base_kernel.get_value().astype(self.dtype))
        if self.base_bias is not None:
            base_out += self.base_bias.get_value().astype(self.dtype)
            
        # lora forward pass
        lora_out = jnp.dot(x, self.lora_A.get_value().astype(self.dtype))
        lora_out = jnp.dot(lora_out, self.lora_B.get_value().astype(self.dtype))
        
        return base_out + lora_out * self.scaling


class DoraLinear(nnx.Module):
    """Weight-Decomposed Low-Rank Adaptation wrapper for nnx.Linear"""
    def __init__(self, base_layer: nnx.Linear, r: int, alpha: int, rngs: nnx.Rngs):
        super().__init__()
        base_w = base_layer.kernel.get_value()
        
        self.base_kernel = nnx.Variable(base_w)
        if base_layer.bias is not None:
            self.base_bias = nnx.Variable(base_layer.bias.get_value())
        else:
            self.base_bias = None

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features, out_features = base_w.shape
        
        # Magnitude vector `m` initialized to the column norms of the pre-trained weight
        # Weight shape: (in_features, out_features)
        base_norm = jnp.linalg.norm(base_w, axis=0, keepdims=True)
        self.m = nnx.Param(base_norm.astype(base_layer.param_dtype))
        
        # A matrix: (in_features, r)
        self.lora_A = nnx.Param(jax.random.normal(rngs.params(), (in_features, r), dtype=base_layer.param_dtype))
        # B matrix: (r, out_features)
        self.lora_B = nnx.Param(jnp.zeros((r, out_features), dtype=base_layer.param_dtype))
        
        self.dtype = base_layer.dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        base_w = self.base_kernel.get_value().astype(self.dtype)
        lora_a = self.lora_A.get_value().astype(self.dtype)
        lora_b = self.lora_B.get_value().astype(self.dtype)
        
        # W' = W + \Delta W
        lora_update = jnp.dot(lora_a, lora_b) * self.scaling
        w_prime = base_w + lora_update
        
        # Normalize the columns of W'
        norm_w_prime = jnp.linalg.norm(w_prime, axis=0, keepdims=True)
        
        # Prevent division by zero
        norm_w_prime = jnp.where(norm_w_prime == 0, 1e-8, norm_w_prime)
        
        # Direction matrix
        direction = w_prime / norm_w_prime
        
        # Final weight = m * direction
        final_w = self.m.get_value().astype(self.dtype) * direction
        
        out = jnp.dot(x, final_w)
        if self.base_bias is not None:
            out += self.base_bias.get_value().astype(self.dtype)
            
        return out

class VeraLinear(nnx.Module):
    """Vector-based Random Adaptation wrapper for nnx.Linear
    Uses frozen random A and B matrices, and trains only the scaling vectors d and b.
    """
    def __init__(self, base_layer: nnx.Linear, r: int, alpha: int, rngs: nnx.Rngs):
        super().__init__()
        self.base_kernel = nnx.Variable(base_layer.kernel.get_value())
        if base_layer.bias is not None:
            self.base_bias = nnx.Variable(base_layer.bias.get_value())
        else:
            self.base_bias = None

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = self.base_kernel.get_value().shape[0]
        out_features = self.base_kernel.get_value().shape[1]
        
        # Frozen A and B matrices (nnx.Variable = not trained)
        rng_A, rng_B = jax.random.split(rngs.params(), 2)
        # Note: In a fully optimized VeRA, these would be shared across all layers.
        # For simplicity in this graph replacement, we instantiate them per layer but keep them frozen.
        self.lora_A = nnx.Variable(jax.random.normal(rng_A, (in_features, r), dtype=base_layer.param_dtype))
        self.lora_B = nnx.Variable(jax.random.normal(rng_B, (r, out_features), dtype=base_layer.param_dtype))
        
        # Trainable scaling vectors (nnx.Param = trained)
        self.d = nnx.Param(jnp.ones((r,), dtype=base_layer.param_dtype))
        self.b = nnx.Param(jnp.ones((out_features,), dtype=base_layer.param_dtype))
        
        self.dtype = base_layer.dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        # base forward
        base_out = jnp.dot(x, self.base_kernel.get_value().astype(self.dtype))
        if self.base_bias is not None:
            base_out += self.base_bias.get_value().astype(self.dtype)
            
        # vera forward
        # x @ A
        lora_out = jnp.dot(x, self.lora_A.get_value().astype(self.dtype))
        # scale by d
        lora_out = lora_out * self.d.get_value().astype(self.dtype)
        # @ B
        lora_out = jnp.dot(lora_out, self.lora_B.get_value().astype(self.dtype))
        # scale by b
        lora_out = lora_out * self.b.get_value().astype(self.dtype)
        
        return base_out + lora_out * self.scaling


class LohaLinear(nnx.Module):
    """Hadamard Product Adaptation (LoHa) wrapper for nnx.Linear"""
    def __init__(self, base_layer: nnx.Linear, r: int, alpha: int, rngs: nnx.Rngs):
        super().__init__()
        self.base_kernel = nnx.Variable(base_layer.kernel.get_value())
        if base_layer.bias is not None:
            self.base_bias = nnx.Variable(base_layer.bias.get_value())
        else:
            self.base_bias = None

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = self.base_kernel.get_value().shape[0]
        out_features = self.base_kernel.get_value().shape[1]
        
        rng_A1, rng_B1, rng_A2, rng_B2 = jax.random.split(rngs.params(), 4)
        
        # LoHa uses four smaller matrices: W_up = (A1 @ B1) \circ (A2 @ B2)
        self.lora_A1 = nnx.Param(jax.random.normal(rng_A1, (in_features, r), dtype=base_layer.param_dtype))
        self.lora_B1 = nnx.Param(jnp.zeros((r, out_features), dtype=base_layer.param_dtype))
        
        self.lora_A2 = nnx.Param(jax.random.normal(rng_A2, (in_features, r), dtype=base_layer.param_dtype))
        self.lora_B2 = nnx.Param(jax.random.normal(rng_A2, (r, out_features), dtype=base_layer.param_dtype))
        
        self.dtype = base_layer.dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        base_out = jnp.dot(x, self.base_kernel.get_value().astype(self.dtype))
        if self.base_bias is not None:
            base_out += self.base_bias.get_value().astype(self.dtype)
            
        a1 = self.lora_A1.get_value().astype(self.dtype)
        b1 = self.lora_B1.get_value().astype(self.dtype)
        a2 = self.lora_A2.get_value().astype(self.dtype)
        b2 = self.lora_B2.get_value().astype(self.dtype)
        
        w1 = jnp.dot(a1, b1)
        w2 = jnp.dot(a2, b2)
        
        loha_update = w1 * w2 # Hadamard product (element-wise multiplication)
        loha_out = jnp.dot(x, loha_update)
        
        return base_out + loha_out * self.scaling


class LokrLinear(nnx.Module):
    """Kronecker Product Adaptation (LoKr) wrapper for nnx.Linear"""
    def __init__(self, base_layer: nnx.Linear, r: int, alpha: int, rngs: nnx.Rngs):
        super().__init__()
        self.base_kernel = nnx.Variable(base_layer.kernel.get_value())
        if base_layer.bias is not None:
            self.base_bias = nnx.Variable(base_layer.bias.get_value())
        else:
            self.base_bias = None

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = self.base_kernel.get_value().shape[0]
        out_features = self.base_kernel.get_value().shape[1]
        
        rng_A1, rng_B1, rng_O = jax.random.split(rngs.params(), 3)
        
        # We factorize the weights as Kronecker product: W_up = Kronecker(A1 @ B1, O)
        # Choosing factorization shapes for demonstration (ideally based on square roots/factors)
        # Assuming r <= min(in_features, out_features), we make the left Kronecker term size (r,r)
        
        # Left Kronecker Term (dynamic LoRA): Shape (r, r)
        self.lora_A = nnx.Param(jax.random.normal(rng_A1, (r, r), dtype=base_layer.param_dtype))
        self.lora_B = nnx.Param(jnp.zeros((r, r), dtype=base_layer.param_dtype))
        
        # Right Kronecker Term (frozen/Random): Shape (in_features // r, out_features // r)
        # We handle padding or reshaping via jnp.kron
        in_factor = max(1, in_features // r)
        out_factor = max(1, out_features // r)
        
        # For a truly generic LoKr, you'd find exact integer factors.
        # Here we just use these factors and slice the Kronecker product later
        self.O = nnx.Param(jax.random.normal(rng_O, (in_factor, out_factor), dtype=base_layer.param_dtype))
        
        self.dtype = base_layer.dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        base_w = self.base_kernel.get_value().astype(self.dtype)
        in_features, out_features = base_w.shape
        
        base_out = jnp.dot(x, base_w)
        if self.base_bias is not None:
            base_out += self.base_bias.get_value().astype(self.dtype)
            
        a = self.lora_A.get_value().astype(self.dtype)
        b = self.lora_B.get_value().astype(self.dtype)
        o_mat = self.O.get_value().astype(self.dtype)
        
        w_left = jnp.dot(a, b)
        
        # Compute Kronecker product
        lokr_update = jnp.kron(w_left, o_mat)
        
        # Since r and factors might not perfectly divide in_features/out_features
        # Slice the Kronecker product down to exact [in_features, out_features] size
        lokr_update = lokr_update[:in_features, :out_features]
        
        lokr_out = jnp.dot(x, lokr_update)
        
        return base_out + lokr_out * self.scaling


class AdaloraLinear(nnx.Module):
    """Adaptive Low-Rank Adaptation (AdaLoRA) wrapper for nnx.Linear.
    Parameterizes the update as P @ E @ Q, where E is a diagonal matrix of singular values.
    During training, E can be dynamically pruned based on importance scores.
    """
    def __init__(self, base_layer: nnx.Linear, r: int, alpha: int, rngs: nnx.Rngs):
        super().__init__()
        self.base_kernel = nnx.Variable(base_layer.kernel.get_value())
        if base_layer.bias is not None:
            self.base_bias = nnx.Variable(base_layer.bias.get_value())
        else:
            self.base_bias = None

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = self.base_kernel.get_value().shape[0]
        out_features = self.base_kernel.get_value().shape[1]
        
        rng_P, rng_Q = jax.random.split(rngs.params(), 2)
        
        # P matrix (Left singular vectors): Shape (in_features, r)
        self.lora_P = nnx.Param(jax.random.normal(rng_P, (in_features, r), dtype=base_layer.param_dtype))
        
        # E vector (Singular values): Shape (r,), initialized to zeros similar to lora_B
        self.lora_E = nnx.Param(jnp.zeros((r,), dtype=base_layer.param_dtype))
        
        # Q matrix (Right singular vectors): Shape (r, out_features)
        self.lora_Q = nnx.Param(jax.random.normal(rng_Q, (r, out_features), dtype=base_layer.param_dtype))
        
        self.dtype = base_layer.dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        base_out = jnp.dot(x, self.base_kernel.get_value().astype(self.dtype))
        if self.base_bias is not None:
            base_out += self.base_bias.get_value().astype(self.dtype)
            
        p = self.lora_P.get_value().astype(self.dtype)
        e = self.lora_E.get_value().astype(self.dtype)
        q = self.lora_Q.get_value().astype(self.dtype)
        
        # AdaLoRA forward: x @ P @ diag(E) @ Q
        # x @ P: (batch, r)
        adalora_out = jnp.dot(x, p)
        # scale by E (broadcasting)
        adalora_out = adalora_out * e
        # @ Q: (batch, out_features)
        adalora_out = jnp.dot(adalora_out, q)
        
        return base_out + adalora_out * self.scaling


def apply_peft(
    model: nnx.Module, 
    method: str = "lora", 
    r: int = 8, 
    alpha: int = 16, 
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
    rngs: nnx.Rngs | None = None
) -> nnx.Module:
    """
    Traverses the model and replaces target nnx.Linear layers with PEFT adapters.
    Modifies the model in-place.
    
    Args:
        model: The base nnx.Module
        method: 'lora', 'dora', 'qlora', etc.
        r: rank
        alpha: scaling factor
        target_modules: list of strings to match layer names against (e.g. ['q_proj', 'v_proj'])
        rngs: nnx.Rngs to initialize new adapter parameters
    """
    if rngs is None:
        rngs = nnx.Rngs(42)
        
    def _replace_recursive(module: nnx.Module, path: str = ""):
        for name, child in vars(module).items():
            if isinstance(child, nnx.Module):
                # Is this a target module that is a Linear layer?
                if any(t in name for t in target_modules) and isinstance(child, nnx.Linear):
                    # Replace!
                    if method.lower() == "lora":
                        adapter = LoraLinear(child, r, alpha, rngs)
                    elif method.lower() == "dora":
                        adapter = DoraLinear(child, r, alpha, rngs)
                    elif method.lower() == "vera":
                        adapter = VeraLinear(child, r, alpha, rngs)
                    elif method.lower() == "loha":
                        adapter = LohaLinear(child, r, alpha, rngs)
                    elif method.lower() == "lokr":
                        adapter = LokrLinear(child, r, alpha, rngs)
                    elif method.lower() == "adalora":
                        adapter = AdaloraLinear(child, r, alpha, rngs)
                    else:
                        raise NotImplementedError(f"PEFT method '{method}' is not yet implemented.")
                    
                    # Overwrite the child node with the adapter node
                    setattr(module, name, adapter)
                else:
                    # Recursive search
                    _replace_recursive(child, path + "." + name)
                    
    _replace_recursive(model)
    return model