# Model

`zlynx.model` contains packaged model architectures and their configs.

See also:

- [Llama](./model-llama.md)
- [DiT](./model-dit.md)
- [SSM](./model-ssm.md)
- [OpenAI](./model-openai.md)
- [Google](./model-google.md)
- [DeepMind](./model-deepmind.md)
- [DeepSeek](./model-deepseek.md)
- [Alibaba](./model-alibaba.md)
- [Moonshot](./model-moonshot.md)
- [Zhipu](./model-zhipu.md)
- [NVIDIA](./model-nvidia.md)
- [Black Forest Labs](./model-blackforestlabs.md)
- [Stability AI](./model-stabilityai.md)

## Root Imports

The root package currently re-exports the stable packaged model classes:

```python
from zlynx.model import LlamaConfig, LlamaLanguageModel, DiTConfig, DiT
```

## Model Packages

Available model packages in the repository include:

- `zlynx.model.llama`
- `zlynx.model.diffusion.dit`
- `zlynx.model.ssm`
- `zlynx.model.deepmind.gemma4`

## Current Public Surface

The model packages are still uneven. In practice:

- `zlynx.model.llama` and `zlynx.model.diffusion.dit` are the current packaged model families
- the org-based placeholder packages exist for future families
- some model trees are still incomplete and should be treated as experimental

If you are building your own model rather than using a packaged architecture, the main base APIs are in [Core](./core.md) and the reusable layers are in [Module](./module.md).
