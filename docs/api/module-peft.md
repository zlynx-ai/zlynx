# Module: PEFT

PEFT adapters live under `zlynx.module.peft`.

## Imports

```python
from zlynx.module.peft import apply_peft, LoraLinear, DoraLinear
```

Available adapter wrappers:

- `LoraLinear`
- `DoraLinear`
- `VeraLinear`
- `LohaLinear`
- `LokrLinear`
- `AdaloraLinear`

## `apply_peft(...)`

```python
from flax import nnx
from zlynx.module.peft import apply_peft

model = apply_peft(
    model,
    method="lora",
    r=8,
    alpha=16,
    target_modules=("q_proj", "v_proj"),
    rngs=nnx.Rngs(0),
)
```

Main arguments:

- `method`
- `r`
- `alpha`
- `target_modules`
- `rngs`

`apply_peft(...)` modifies the model in place and replaces matching `nnx.Linear` layers with adapter wrappers.

For workflow-level usage, see [PEFT](../useful-stuff/peft.md).
