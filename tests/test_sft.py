import pytest
import jax
import jax.numpy as jnp
from zlynx.model.llama import LlamaConfig, LlamaLanguageModel
from zlynx.trainer.sft import SFTTrainer, SFTConfig
from zlynx.trainer.dataset import DatasetConfig

# Fake Processor class to mock a Hugging Face Tokenizer
class DummyProcessor:
    def __init__(self, vocab_size, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
    def __call__(self, text, padding, truncation, max_length, return_tensors):
        # Mocks taking a string and returning padded numpy arrays
        import numpy as np
        if isinstance(text, str):
            text = [text]
            
        input_ids = []
        attention_mask = []
        
        for t in text:
            # dummy naive encoding: length of string dictates tokens
            length = min(len(t.split()), max_length)
            tokens = np.random.randint(1, self.vocab_size, size=length)
            
            # pad
            padded_tokens = np.pad(tokens, (0, max_length - length), constant_values=self.pad_token_id)
            mask = np.pad(np.ones(length), (0, max_length - length), constant_values=0)
            
            input_ids.append(padded_tokens)
            attention_mask.append(mask)
            
        return {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "attention_mask": np.array(attention_mask, dtype=np.int32)
        }

def test_sft_trainer_initialization_and_formatting():
    print("\n=== Testing SFTTrainer formatting pipeline ===")
    config = LlamaConfig(vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=1, head_dim=8)
    model = LlamaLanguageModel(config, key=jax.random.key(123))
    processor = DummyProcessor(vocab_size=16)
    
    # Dummy raw text dataset (like HF dataset format)
    raw_dataset = [
        {"text": "hello this is a test string for SFT"},
        {"text": "another sequence"},
        {"text": "short"},
        {"text": "a very very very very extraordinarily long sequence to test truncation"}
    ]
    
    # 1. Initialize SFTConfig overriding the seq_len
    sft_config = SFTConfig(
        max_seq_len=8,
        batch_size=2,
        max_steps=2,
        learning_rate=1e-3,
        logging_steps=1,
        sharding=0 # single device local test
    )
    
    ds_config = DatasetConfig(shuffle=False)
    
    # 2. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        dataset=raw_dataset,
        processor=processor,
        trconfig=sft_config,
        dsconfig=ds_config
    )
    
    # 3. Test that the dataset was properly mapped
    # It should have converted {"text": ...} into {"input_ids": [8], "attention_mask": [8]}
    # The dataset_iter is created inside Trainer.train()
    
    # Grab one batch manually for validation
    ds_iter = iter(trainer.dataset)
    batch = next(ds_iter)
    
    # Verify shape [batch_size, max_seq_len]
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    
    # Verify values exist
    assert jnp.any(batch["attention_mask"] == 1)
    
    print("SFT Dataset Mapping OK!")
    
    # 4. Run short training loop to verify causal_lm_loss computes gradients successfully
    print("\n=== Testing SFTTrainer training execution ===")
    trainer.train()
    print("SFT Training executed successfully!")

if __name__ == "__main__":
    test_sft_trainer_initialization_and_formatting()
