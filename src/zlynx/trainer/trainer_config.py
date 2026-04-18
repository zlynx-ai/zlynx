

from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Literal

@dataclass
class TrainerConfig:
    # ── jax ──
    jit_loss_fn: bool = True

    # ── batch / accumulation ──
    # `batch_size` is the number of samples fed into **one model replica** per
    # micro-step. With sharding="ddp" on N devices there are N replicas, so the
    # effective batch becomes batch_size × N × gradient_accumulation_steps.
    # With "fsdp"/"tp"/"pp"/single-device there is only 1 replica (the model is
    # sharded, not replicated), so effective batch = batch_size × gradient_accumulation_steps.
    # For sharding=None (custom user-managed mesh) the replica count is unknown
    # and the trainer does not attempt to infer it.
    batch_size: int = 1
    gradient_accumulation_steps: int = 1  # effective batch = batch_size × replicas × this

    # ── dataset ──
    # ── use when specify train_dataset/eval_dataset as str ──
    train_subset: str = "default"
    eval_subset: str = "default"
    train_split: str = "train"
    eval_split: str = "test"
    streaming: bool = False 
    train_streaming: Optional[bool] = None  # None = will use from `streaming`
    eval_streaming: Optional[bool] = None   # None = will use from `streaming`
    load_train_dataset_fn: Optional[Callable] = None
    load_eval_dataset_fn: Optional[Callable] = None

    # ── process dataset ──
    processing_train_dataset_fn: Optional[Callable] = None
    processing_eval_dataset_fn: Optional[Callable] = None
    seed: Optional[int] = None

    # ── grain ──
    num_workers: int = 4
    num_threads: int = 16
    prefetch_buffer_size: int = 1_000

    # ── optimizer ──
    optimizer: str | Callable = "adamw"                     # "adamw", "sgd", ...
    optimizer_kwargs: dict = field(default_factory=dict)  # extra kwargs forwarded to optax
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0            # gradient clipping (None = disabled)

    # ── schedule ──
    lr_scheduler: str | Callable = "cosine"      # callable args lr, total, warmup
    warmup_steps: int = 0
    warmup_ratio: float = 0.0                    # alternative: fraction of total steps

    # ── training duration ──
    max_steps: int = -1                          # -1 = use num_epochs instead
    num_epochs: int = 1

    # ── evaluation ──
    # if `eval_strategy = "epochs"` -> use eval_epochs
    # if `eval_strategy = "step"` -> use eval_steps
    eval_strategy: Literal["epochs", "steps"] = "epochs"
    eval_max_steps: int = -1                        # -1 = whole eval_dataset but needed > 0 for `eval_streaming = True`
    eval_steps: int | None = None                   # None = eval every epoch if there is `eval_dataset`
    eval_epochs: int | None = None                  # None = eval every epoch if there is `eval_dataset`
    eval_batch_size: int | None = None   # None = same as batch_size

    # ── precision ──
    remat: bool = False                          # recompute activations to save memory (jax.remat)

    # ── sharding / parallelism ──
    sharding: str | int | bool | None = "auto"     # "auto", None (skip), "fsdp", "tp", "dp", int (device ID), False (single)

    # ── checkpointing ──
    output_dir: str = "./output"
    save_steps: int = 500
    save_total_limit: int | None = 3             # max checkpoints to keep (None = unlimited)
    resume_from: str | None = None               # checkpoint path to resume from

    # ── logging ──
    logging_steps: int = 10
    log_to: list[str] = field(default_factory=list)  # ["wandb", "tensorboard"]
    run_name: str | None = None
    logging_fn: Dict[str, Callable] | None = None  # custom metrics: {"perplexity": lambda **kw: jnp.exp(kw["loss"])}
    logging_aux: bool = True                       # log aux return from loss function immediately

    # Optional custom optimizer builder.
    #
    # If provided, this function overrides the default optimizer construction logic
    # in `zlynx.trainer.optim.build_optimizer`.
    #
    # This is useful when:
    # - a custom optimizer requires special initialization,
    # - the optimizer/scheduler cannot be constructed by the default builder,
    # - or you want full control over how the optimizer and learning-rate schedule are created.
    #
    # Signature:
    #     build_optimizer_fn(config, total_steps) -> tuple[optax.GradientTransformation, schedule]
    #
    # Args:
    #     config:
    #         The current trainer configuration.
    #     total_steps:
    #         The total number of optimization steps used to build the scheduler.
    #         Usually this is `max_steps`.
    #         If `max_steps == -1`, it will be recomputed from the dataset size and
    #         training configuration during `Trainer.train()`.
    #
    # Returns:
    #     A tuple `(optimizer, schedule)`.
    #
    # Example:
    #     def build_optimizer_fn(config, total_steps):
    #         ...
    #         return opt, schedule
    build_optimizer_fn: Optional[Callable] = None


    # Optional hook for providing extra keyword arguments to `optimizer.update(...)`.
    #
    # Some optimizers may require additional values during update, such as
    # `value`, `grad`, `value_fn`, `grad_fn`, or other optimizer-specific arguments.
    # This hook allows users to customize or correct those values before they are
    # passed into the optimizer.
    #
    # Signature:
    #     return_optimizer_extra_args_fn(
    #         value,
    #         grad,
    #         value_fn,
    #         grad_fn,
    #         compute_loss_and_grads_fn,
    #     ) -> dict
    #
    # Args:
    #     value:
    #         The scalar loss value corresponding to the current optimization update.
    #         When gradient accumulation is enabled, this should normally correspond
    #         to the accumulated/averaged loss used for that update step.
    #
    #     grad:
    #         The gradients used for the optimizer step.
    #         When gradient accumulation is enabled, this is typically the final
    #         accumulated gradient tensor tree for the current update.
    #
    #     value_fn:
    #         A helper function derived from `compute_loss_and_grads_fn` that returns
    #         only the loss value.
    #
    #     grad_fn:
    #         A helper function derived from `compute_loss_and_grads_fn` that returns
    #         only the gradients.
    #
    #     compute_loss_and_grads_fn:
    #         A function with signature `(model, loss_fn, batch)` that returns:
    #             `(loss, aux, grads)`
    #         where:
    #             - `loss` is the scalar training loss,
    #             - `aux` is the auxiliary output returned by the user-defined `loss_fn`,
    #             - `grads` is the gradient tree.
    #
    # Returns:
    #     A dictionary of extra keyword arguments that will be forwarded to
    #     `optimizer.update(...)`.
    #
    # Example:
    #     def return_optimizer_extra_args_fn(value, grad, value_fn, grad_fn, compute_loss_and_grads_fn):
    #         return {
    #             "value": value,
    #             "grad": grad,
    #             "value_fn": value_fn,
    #             "grad_fn": grad_fn,
    #         }
    return_optimizer_extra_args_fn: Optional[Callable] = None

# @dataclass
# class SFTConfig(TrainerConfig):
#     """Configuration for Supervised Fine-Tuning (SFT)."""
#     max_seq_len: int = 1024
#     dataset_text_field: str = "text"
#     formatting_func: Optional[Callable] = None


# @dataclass
# class GRPOConfig(TrainerConfig):
#     """Configuration for Group Relative Policy Optimization (GRPO)."""
#     max_seq_len: int = 1024
#     num_generations: int = 8               # G = Group size
#     max_prompt_length: int = 512
#     max_completion_length: int = 512       # Ensure prompt + compl = max_seq_len
#     beta: float = 0.0                      # KL Divergence penalty
#     clip_eps: float = 0.2                  # PPO clipping ratio
#     entropy_coeff: float = 0.0             # Entropy bonus coefficient
#     mu_epochs: int = 1                     # Inner optimization epochs per generation
#     dataset_prompt_field: str = "prompt"
#     dataset_responses_field: str = "responses"


# @dataclass
# class DSFTConfig(TrainerConfig):
#     """Configuration for Diffusion Supervised Fine-Tuning (DSFT)."""
#     image_size: tuple[int, int] = (256, 256)
#     dataset_image_field: str = "image"
#     dataset_text_field: str = "text"
#     formatting_func: Optional[Callable] = None
    
#     # Diffusion specific
#     num_train_timesteps: int = 1000
#     prediction_type: str = "epsilon" # "epsilon", "v_prediction", or "sample"


# @dataclass
# class DPOConfig(TrainerConfig):
#     """Configuration for Direct Preference Optimization (DPO)."""
#     beta: float = 0.1
#     max_prompt_length: int = 512
#     max_seq_len: int = 1024
#     dataset_prompt_field: str = "prompt"
#     dataset_chosen_field: str = "chosen"
#     dataset_rejected_field: str = "rejected"
#     loss_type: str = "sigmoid" # "sigmoid", "hinge", "ipo", "kto_pair"
#     label_smoothing: float = 0.0
