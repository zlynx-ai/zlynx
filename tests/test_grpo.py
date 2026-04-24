import pytest
import jax
import jax.numpy as jnp
from zlynx.model.meta.llama import LlamaConfig, LlamaLanguageModel
from zlynx.trainer.grpo import GRPOTrainer, GRPOConfig
import numpy as np

# Fake Processor class to mock a Hugging Face Tokenizer
class DummyProcessor:
    def __init__(self, vocab_size, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
    def __call__(self, text, padding, truncation, max_length, return_tensors):
        # Mocks taking a string and returning padded numpy arrays
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
        
    def decode(self, token_ids, skip_special_tokens=True):
        return "This is a dummy decoded string correct answer"

def test_grpo_trainer_execution():
    print("\n=== Testing GRPOTrainer Rollouts ===")
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, head_dim=8)
    model = LlamaLanguageModel(config, key=jax.random.key(2026))
    processor = DummyProcessor(vocab_size=64)
    
    dummy_dataset = [
        {
            "input_ids": np.ones((8,), dtype=np.int32),
            "attention_mask": np.ones((8,), dtype=np.int32)
        }
        for _ in range(4)
    ]

    # Custom reward function
    def my_reward(completion: str) -> float:
        if "correct" in completion.lower():
            return 2.5
        return -1.0

    grpo_config = GRPOConfig(
        max_prompt_length=8,
        max_seq_len=16,          # Leaves 8 tokens for max_new_tokens
        num_generations=2,       # G=2
        batch_size=1,
        max_steps=2,             # Run exactly 2 steps
        learning_rate=1e-3,
        logging_steps=1,
        mu_epochs=2,             # 2 inner PPO optimization loops per generation
    )
    
    trainer = GRPOTrainer(
        model=model,
        ref_model=None, # Testing purely entropy + rewards without KL constraint
        reward_funcs=[my_reward],
        dataset=dummy_dataset,
        processor=processor,
        trconfig=grpo_config,
    )
    
    print("Running trainer.train()...")
    trainer.train()
    print("Trainer finished seamlessly!")
