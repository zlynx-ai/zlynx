import os
import jax
import jax.numpy as jnp
from zlynx.model.llama import LlamaConfig, LlamaLanguageModel

def test_load_mixed_precision():
    # 1. Create a model with bf16 for both and save it
    print("Creating base model (bf16)")
    config = LlamaConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, head_dim=8,
        dtype="bfloat16", param_dtype="bfloat16"
    )
    model = LlamaLanguageModel(config)
    model.save("./tmp_load_test")

    # 2. Check the dtypes:
    print(f"Original compute dtype: {model.model.blocks[0].mlp.gate_proj.dtype}")
    print(f"Original param dtype: {model.model.blocks[0].mlp.gate_proj.kernel.get_value().dtype}")
    print("-" * 50)

    # 3. Load config, modify param_dtype to fp32, and load model
    print("Loading config and changing param_dtype to fp32")
    loaded_config = LlamaLanguageModel.load_config("./tmp_load_test")
    modified_config = loaded_config.replace(param_dtype="float32")

    # 4. Load model with modified config
    # bypass tokenizer load since we didn't save one
    LlamaLanguageModel.processor = None
    loaded_model, processor = LlamaLanguageModel.load("./tmp_load_test", config=modified_config)

    # 5. Verify the dtypes
    print(f"Loaded compute dtype: {loaded_model.model.blocks[0].mlp.gate_proj.dtype}")
    print(f"Loaded param dtype: {loaded_model.model.blocks[0].mlp.gate_proj.kernel.get_value().dtype}")

    # 6. Ensure they are actually different
    assert loaded_model.model.blocks[0].mlp.gate_proj.dtype == jnp.bfloat16
    assert loaded_model.model.blocks[0].mlp.gate_proj.kernel.get_value().dtype == jnp.float32

    print("SUCCESS: Mixed precision config works! param_dtype is fp32 while compute dtype is bf16.")

if __name__ == "__main__":
    test_load_mixed_precision()
