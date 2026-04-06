# Your First Model

Learn how to define neural networks with Zlynx using the `Z` base class and Flax NNX layers.

---

## The `Z` Base Class

Every Zlynx model inherits from `Z`, which itself extends Flax's `nnx.Module`. This gives you:

- **`model.save(path)`** — save weights to disk via Orbax
- **`Model.load(path, ...)`** — reconstruct and load weights back
- All standard `nnx.Module` features (parameter tracking, JIT compatibility, etc.)

> [!IMPORTANT]
> Only the **outermost** model needs to inherit from `Z`. Inner layers like `nnx.Conv` and `nnx.Linear` are standard Flax modules.

---

## A Simple MLP

Let's start with the simplest possible model — a multi-layer perceptron:

```python
import jax
import jax.numpy as jnp
from flax import nnx
from zlynx import Z


class MLP(Z):
    def __init__(self, key, in_features: int, hidden: int, out_features: int):
        super().__init__()
        k1, k2 = jax.random.split(key)

        self.linear1 = nnx.Linear(in_features, hidden, rngs=nnx.Rngs(k1))
        self.linear2 = nnx.Linear(hidden, out_features, rngs=nnx.Rngs(k2))

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)
```

### Key patterns:

1. **`super().__init__()`** — always call this first
2. **Random keys** — JAX uses explicit PRNG keys. Split a parent key into sub-keys, one per layer
3. **`rngs=nnx.Rngs(key)`** — wraps a JAX key for Flax NNX layers
4. **`__call__`** — defines the forward pass. Returns raw output (logits, features, etc.)

### Using the model:

```python
key = jax.random.key(42)
model = MLP(key, in_features=784, hidden=256, out_features=10)

# Test with random input
x = jax.random.normal(jax.random.key(0), (4, 784))   # batch of 4
logits = model(x)                                       # (4, 10)
print(f"Output shape: {logits.shape}")
```

---

## A CNN for Image Classification

For images, use `nnx.Conv` and `nnx.max_pool`:

```python
class CNN(Z):
    def __init__(self, key, num_classes: int = 10):
        super().__init__()
        k1, k2, k3 = jax.random.split(key, 3)

        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding="VALID", rngs=nnx.Rngs(k1))
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="VALID", rngs=nnx.Rngs(k2))
        self.fc    = nnx.Linear(64 * 5 * 5, num_classes, rngs=nnx.Rngs(k3))

    def __call__(self, x):
        # x: (batch, height, width, channels)
        x = jax.nn.relu(self.conv1(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = jax.nn.relu(self.conv2(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape(x.shape[0], -1)   # flatten spatial dims
        return self.fc(x)
```

```python
key = jax.random.key(42)
model = CNN(key, num_classes=10)

# MNIST-shaped input: (batch, 28, 28, 1)
x = jax.random.normal(jax.random.key(0), (4, 28, 28, 1))
logits = model(x)   # (4, 10)
print(f"Output shape: {logits.shape}")
```

---

## Building Blocks Cheat Sheet

Common Flax NNX layers you can use inside your `Z` model:

| Layer           | Signature                                            | Notes                                       |
| --------------- | ---------------------------------------------------- | ------------------------------------------- |
| `nnx.Linear`    | `(in_features, out_features, rngs=...)`              | Fully connected                             |
| `nnx.Conv`      | `(in_features, out_features, kernel_size, rngs=...)` | Convolution (supports `padding`, `strides`) |
| `nnx.Embed`     | `(num_embeddings, features, rngs=...)`               | Embedding lookup table                      |
| `nnx.BatchNorm` | `(num_features, rngs=...)`                           | Batch normalization                         |
| `nnx.LayerNorm` | `(num_features)`                                     | Layer normalization                         |
| `nnx.Dropout`   | `(rate, rngs=...)`                                   | Dropout (stochastic during training)        |

Activation functions live in `jax.nn`:

```python
jax.nn.relu(x)
jax.nn.gelu(x)
jax.nn.silu(x)      # also called swish
jax.nn.softmax(x)
```

Pooling lives in `nnx`:

```python
nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
```

---

## Composing Sub-Modules

You can nest `Z` or `nnx.Module` subclasses freely:

```python
class ResidualBlock(nnx.Module):
    """A simple residual block — no need to inherit Z here."""
    def __init__(self, features: int, key):
        k1, k2 = jax.random.split(key)
        self.linear1 = nnx.Linear(features, features, rngs=nnx.Rngs(k1))
        self.linear2 = nnx.Linear(features, features, rngs=nnx.Rngs(k2))

    def __call__(self, x):
        residual = x
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        return x + residual


class ResNet(Z):
    """Only the top-level model inherits Z."""
    def __init__(self, key, features: int = 256, num_blocks: int = 4, num_classes: int = 10):
        super().__init__()
        keys = jax.random.split(key, num_blocks + 1)

        self.input_proj = nnx.Linear(784, features, rngs=nnx.Rngs(keys[0]))
        self.blocks = [ResidualBlock(features, keys[i + 1]) for i in range(num_blocks)]
        self.head = nnx.Linear(features, num_classes, rngs=nnx.Rngs(keys[-1]))

    def __call__(self, x):
        x = jax.nn.relu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.head(x)
```

> [!TIP]
> Python lists of sub-modules work fine — Flax NNX automatically tracks all parameters nested inside the model tree.

---

## Next Steps

Your model is ready! Head to [Training](./03_training.md) to learn how to train it with Zlynx's `Trainer`.
