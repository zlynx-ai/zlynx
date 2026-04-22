# Installation

Get Zlynx running in under a minute.

## Quick Install

Zlynx is a Python package built on [JAX](https://github.com/jax-ml/jax) and [Flax NNX](https://github.com/google/flax). It requires **Python 3.12+**.

### With uv

```bash
uv pip install zlynx
```

### With pip

```bash
pip install zlynx
```

This installs the default CPU-friendly setup.

## Verify Your Installation

Run a quick check to confirm everything works:

```python
import jax
import zlynx

print(f"JAX version:    {jax.__version__}")
print(f"zlynx version:  {zlynx.__version__}")
print(f"JAX backend:    {jax.default_backend()}")
print(f"Devices:        {jax.devices()}")
```

You should see output like:

```
JAX version:    0.9.2
zlynx version:  0.1.10
JAX backend:    cpu        ← or "gpu" / "tpu"
Devices:        [CpuDevice(id=0)]
```

## Backend Extras

ZLynx exposes JAX backend extras so you can select the install target directly from the package name.

### CPU

```bash
pip install "zlynx[cpu]"
```

This is effectively the same as:

```bash
pip install zlynx
```

### NVIDIA GPU (CUDA)

```bash
pip install "zlynx[cuda]"
pip install "zlynx[cuda12]"
pip install "zlynx[cuda13]"
```

### NVIDIA GPU with local CUDA installation

```bash
pip install "zlynx[cuda12-local]"
pip install "zlynx[cuda13-local]"
```

Use the `-local` variants when CUDA is already installed on the system and you do not want pip-managed CUDA runtime wheels.

### Google Cloud TPU

```bash
pip install "zlynx[tpu]"
```

## Backend Notes

- `cuda` is a convenience alias matching JAX's default CUDA extra
- `cuda12` and `cuda13` give explicit CUDA version targets
- `tpu` is intended for Cloud TPU VM environments
- ROCm is not currently exposed as a ZLynx extra because the upstream JAX ROCm packaging is not yet stable enough for a clean default experience

If you need a backend configuration that is not covered by the ZLynx extras, install ZLynx first and then install the desired JAX variant:

```bash
pip install zlynx
pip install -U "jax[...]"
```

After installing your chosen backend, verify with:

```python
import jax
print(jax.default_backend())   # Should print "gpu" or "tpu"
print(jax.devices())            # Should list your accelerator(s)
```

## Next Steps

You're all set. Head to [Create a model](./create-a-model.md) to build model with ZLynx.
