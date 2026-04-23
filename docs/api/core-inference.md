# Core: Inference

`zlynx.core.inferences` contains the language-model generation helper.

## Main Class

```python
from zlynx.core import LanguageModel
```

`LanguageModel` extends `PretrainedModel` and provides:

- `generate(...)`

## `generate(...)`

The current generation helper supports common decoding controls such as:

- `max_new_tokens`
- `temperature`
- `top_k`
- `top_p`
- `repetition_penalty`
- `eos_token_id`
- `suppress_tokens`

It expects a language-model style forward pass that can return:

- `logits`
- optional `past_key_values`

## Notes

- `temperature=0.0` gives greedy decoding.
- KV cache support depends on the model implementation.
- this is a helper for autoregressive language models, not a generic inference framework.
