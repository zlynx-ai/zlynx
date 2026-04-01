

from dataclasses import dataclass, field
from typing import Callable, Optional



@dataclass
class DatasetConfig:
    # ── source ──
    path: str | None = None                      # HF dataset name or local path
    subset: str | None = None                    # dataset config/subset name
    split: str = "train"
    eval_split: str | None = None                # e.g. "validation"
    streaming: bool = False
    force_streaming: bool = False

    # ── preprocessing ──
    preprocessing_fn: Callable | None = None
    filter_fn: Callable | None = None

    # ── shuffling / sampling ──
    shuffle: bool = True
    shuffle_seed: int = 42

    # ── performance ──
    num_workers: int = 4
    num_threads: int = 16
    prefetch_buffer_size: int = 1_000


@dataclass
class TrainerConfig:
    # ── batch / accumulation ──
    batch_size: int = 1
    gradient_accumulation_steps: int = 1  # effective batch = batch_size × this

    # ── optimizer ──
    optimizer: str = "adamw"                     # "adamw", "sgd", "lion", "muon", ...
    optimizer_kwargs: dict = field(default_factory=dict)  # extra kwargs forwarded to optax
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0            # gradient clipping (None = disabled)

    # ── schedule ──
    lr_scheduler: str = "cosine"                 # "cosine", "linear", "constant", "warmup_cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0                    # alternative: fraction of total steps

    # ── training duration ──
    max_steps: int = -1                          # -1 = use num_epochs instead
    num_epochs: int = 1

    # ── precision ──
    dtype: str = "bfloat16"                      # param / compute dtype
    grad_dtype: str | None = None                # gradient dtype (None = same as dtype)
    gradient_checkpointing: bool = False         # recompute activations to save memory (jax.remat)

    # ── sharding / parallelism ──
    sharding: str | int | bool | None = "auto"     # "auto", None (skip), "fsdp", "tp", "dp", int (device ID), False (single)
    mesh_shape: tuple[int, ...] | None = None    # custom device mesh shape

    # ── checkpointing ──
    output_dir: str = "./output"
    save_steps: int = 500
    save_total_limit: int | None = 3             # max checkpoints to keep (None = unlimited)
    resume_from: str | None = None               # checkpoint path to resume from

    # ── logging ──
    logging_steps: int = 10
    log_to: list[str] = field(default_factory=lambda: ["stdout"])  # ["stdout", "wandb", "tensorboard", "json"]
    run_name: str | None = None
    logging_fn: dict[str, Callable] | None = None  # custom metrics: {"perplexity": lambda **kw: jnp.exp(kw["loss"])}

    # ── evaluation ──
    eval_steps: int | None = None                # None = no eval during training
    eval_batch_size: int | None = None           # None = same as batch_size


@dataclass
class SFTConfig(TrainerConfig):
    """Configuration for Supervised Fine-Tuning (SFT)."""
    max_seq_len: int = 1024
    dataset_text_field: str = "text"
    formatting_func: Optional[Callable] = None


@dataclass
class GRPOConfig(TrainerConfig):
    """Configuration for Group Relative Policy Optimization (GRPO)."""
    max_seq_len: int = 1024
    num_generations: int = 8               # G = Group size
    max_prompt_length: int = 512
    max_completion_length: int = 512       # Ensure prompt + compl = max_seq_len
    beta: float = 0.0                      # KL Divergence penalty
    clip_eps: float = 0.2                  # PPO clipping ratio
    entropy_coeff: float = 0.0             # Entropy bonus coefficient
    mu_epochs: int = 1                     # Inner optimization epochs per generation
    dataset_prompt_field: str = "prompt"
    dataset_responses_field: str = "responses"


@dataclass
class DSFTConfig(TrainerConfig):
    """Configuration for Diffusion Supervised Fine-Tuning (DSFT)."""
    image_size: tuple[int, int] = (256, 256)
    dataset_image_field: str = "image"
    dataset_text_field: str = "text"
    formatting_func: Optional[Callable] = None
    
    # Diffusion specific
    num_train_timesteps: int = 1000
    prediction_type: str = "epsilon" # "epsilon", "v_prediction", or "sample"


@dataclass
class DPOConfig(TrainerConfig):
    """Configuration for Direct Preference Optimization (DPO)."""
    beta: float = 0.1
    max_prompt_length: int = 512
    max_seq_len: int = 1024
    dataset_prompt_field: str = "prompt"
    dataset_chosen_field: str = "chosen"
    dataset_rejected_field: str = "rejected"
    loss_type: str = "sigmoid" # "sigmoid", "hinge", "ipo", "kto_pair"
    label_smoothing: float = 0.0
