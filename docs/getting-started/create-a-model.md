# Create a Model

Define neural networks with Zlynx using Flax NNX layers.

You can build models in two ways:

- use a plain `nnx.Module` if you only need forward/training behavior
- use `Z` if you also want built-in save/load and hub helpers

Both work with ZLynx's `Trainer`.

## When to Use `Z`

`Z` is ZLynx's model base class. It extends `nnx.Module` and adds model artifact utilities such as:

- `model.save(path)`
- `Model.load(path, ...)`
- `model.push_hf(...)`
- `Model.load_hf(...)`
- `model.push_kaggle(...)`
- `Model.load_kaggle(...)`

If you only want to train a model and do not need those artifact helpers yet, a plain `nnx.Module` is enough.

> [!IMPORTANT]
> You do not need to inherit from Z to use Trainer. But if you choose to use Z for save/load
> and hub helpers, only the outermost model should inherit from it. Inner layers and
> submodules should remain standard nnx.Module classes.

## A Simple MLP

Here is a minimal model using `Z`:

```python
import jax
from flax import nnx
from zlynx import Z


class MLP(Z):
    def __init__(self, rngs: nnx.Rngs, in_features: int, hidden: int, out_features: int):
        self.linear1 = nnx.Linear(in_features, hidden, rngs=rngs)
        self.linear2 = nnx.Linear(hidden, out_features, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)
```

### Key Patterns

1. **Assign submodules in `__init__`** — layers become part of the tracked model tree
2. **`nnx.Rngs(...)`** — Flax NNX manages PRNG streams with internal counters
3. **Pass the same `rngs` object down** — submodules can draw from it without manual key splitting
4. **`__call__`** — defines the forward pass. Returns raw output (logits, features, etc.)

If you want the underlying Flax NNX behavior in more detail, see the official
[`flax.nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html)
and
[`flax.nnx.Rngs`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/rnglib.html)
references.

### Using the model

```python
rngs = nnx.Rngs(42)
model = MLP(rngs, in_features=784, hidden=256, out_features=10)

# Test with random input
x = jax.random.normal(jax.random.key(0), (4, 784))   # batch of 4
logits = model(x)                                       # (4, 10)
print(f"Output shape: {logits.shape}")
```

## Plain `nnx.Module` Also Works

If you do not need save/load helpers yet, you can define the same kind of model with plain NNX:

```python
import jax
from flax import nnx

class PlainMLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, in_features: int, hidden: int, out_features: int):
        self.linear1 = nnx.Linear(in_features, hidden, rngs=rngs)
        self.linear2 = nnx.Linear(hidden, out_features, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)
```

This is fully usable with `Trainer`. The main thing you do not get is `Z`'s built-in checkpoint and hub API.

For the underlying module API, see the official
[`flax.nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html)
reference.

## A CNN for Image Classification

For images, use `nnx.Conv` and `nnx.max_pool`:

```python
class CNN(Z):
    def __init__(self, rngs: nnx.Rngs, num_classes: int = 10):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding="VALID", rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="VALID", rngs=rngs)
        self.fc    = nnx.Linear(64 * 5 * 5, num_classes, rngs=rngs)

    def __call__(self, x):
        # x: (batch, height, width, channels)
        x = jax.nn.relu(self.conv1(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = jax.nn.relu(self.conv2(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape(x.shape[0], -1)   # flatten spatial dims
        return self.fc(x)
```

This follows the same structure as the MLP example:

- define NNX layers in `__init__`
- apply them in `__call__`
- flatten before the final linear classifier

```python
rngs = nnx.Rngs(42)
model = CNN(rngs, num_classes=10)

# MNIST-shaped input: (batch, 28, 28, 1)
x = jax.random.normal(jax.random.key(0), (4, 28, 28, 1))
logits = model(x)   # (4, 10)
print(f"Output shape: {logits.shape}")
```

## Building Blocks Cheat Sheet

Common Flax NNX layers you can use inside your model:

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

For more layer APIs, see the official
[`flax.nnx`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/index.html) layer references.

## Composing Sub-Modules

You can nest `Z` and `nnx.Module` subclasses freely:

```python
class ResidualBlock(nnx.Module):
    def __init__(self, features: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(features, features, rngs=rngs)
        self.linear2 = nnx.Linear(features, features, rngs=rngs)

    def __call__(self, x):
        residual = x
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        return x + residual


class ResNet(Z):
    def __init__(self, rngs: nnx.Rngs, features: int = 256, num_blocks: int = 4, num_classes: int = 10):
        self.input_proj = nnx.Linear(784, features, rngs=rngs)
        self.blocks = nnx.List([ResidualBlock(features, rngs) for _ in range(num_blocks)])
        self.head = nnx.Linear(features, num_classes, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.relu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.head(x)
```

> [!TIP]
> For repeated submodules, use `nnx.List(...)` rather than a plain Python list so NNX tracks the collection correctly.
> See the official
> [`flax.nnx.helpers`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/helpers.html) helpers reference
> for more on tracked helper containers.

## Next Steps

Once your model is defined, continue with [Training](./training.md).
