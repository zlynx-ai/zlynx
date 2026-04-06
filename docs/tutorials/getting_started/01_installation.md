# Installation

Get Zlynx running in under a minute.

---

## Quick Install

Zlynx is a Python package built on [JAX](https://github.com/jax-ml/jax) and [Flax NNX](https://github.com/google/flax). It requires **Python 3.12+**.

### With uv (recommended)

```bash
uv pip install zlynx
```

### With pip

```bash
pip install zlynx
```

---

## Verify Your Installation

Run a quick check to confirm everything works:

```python
import jax
import zlynx

print(f"JAX version:    {jax.__version__}")
print(f"JAX backend:    {jax.default_backend()}")
print(f"Devices:        {jax.devices()}")
```

You should see output like:

```
JAX version:    0.9.1
JAX backend:    cpu        ← or "gpu" / "tpu"
Devices:        [CpuDevice(id=0)]
```

---

## GPU / TPU Setup

By default JAX runs on CPU. To use accelerators, install the appropriate JAX variant **before** installing Zlynx:

### NVIDIA GPU (CUDA)

```bash
pip install -U "jax[cuda12]"
```

### Google Cloud TPU

```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Apple Silicon (Metal)

```bash
pip install -U jax-metal
```

After installing, verify with:

```python
import jax
print(jax.default_backend())   # Should print "gpu" or "tpu"
print(jax.devices())            # Should list your accelerator(s)
```

---

## Dependencies

Zlynx automatically installs these core dependencies:

| Package      | Purpose                                  |
| ------------ | ---------------------------------------- |
| `jax`        | XLA-accelerated numerical computing      |
| `flax`       | Neural network building blocks (NNX API) |
| `optax`      | Optimizers and learning rate schedules   |
| `orbax`      | Checkpointing (save/load model weights)  |
| `grain`      | High-performance data loading pipeline   |
| `datasets`   | Hugging Face dataset loading             |
| `tokenizers` | Fast tokenization                        |

---

## Next Steps

You're all set! Head to [Your First Model](./02_your_first_model.md) to build and run a neural network with Zlynx.
