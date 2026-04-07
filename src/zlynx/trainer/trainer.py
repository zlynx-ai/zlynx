
from pathlib import Path
import time
import jax, jax.numpy as jnp
from flax import nnx
from orbax import checkpoint as ocp
from typing import Callable, Optional, Any
import optax
from tqdm import tqdm
import sys

from .trainer_config import TrainerConfig
from .loss_fn import (
    causal_lm_loss, compute_loss_and_grads, 
    grpo_loss_fn, dpo_loss_fn, 
    token_log_probs, diffusion_loss
)
from .process import process_model, process_dataset, apply_remat
from .optim import build_optimizer, find_galore_state, set_galore_state, update_galore_projectors
from .logger import Logger, TrainingMetric, print_training_banner, _get_training_banner_args

# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model,
        loss_fn: Callable,
        train_dataset,
        eval_dataset: Optional[Any] = None,
        config: TrainerConfig | None = None,
    ):
        """
        Initialize the trainer with a model, loss function, datasets, and config.

        Args:
            model:
                The model to train. This same model is passed to `loss_fn`
                during training and evaluation.

            loss_fn:
                A callable with signature `loss_fn(model, batch)`.

                - `model` is the model passed into the trainer.
                - `batch` is a batch of data produced from the dataset pipeline,
                  already grouped using the batch size defined in `config`.

                The function may return either:

                1. A loss directly: `loss`

                2. A dictionary containing at least `"loss"`:
                   ```
                   {"loss": loss, ...}
                   ```

                Returning a dict allows extra values to be exposed to
                `config.logging_fn`. In that case, each callable in
                `config.logging_fn` receives the returned dictionary as
                keyword arguments.

                Example:
                    ```
                    {
                        "loss": loss, 
                        "nll": nll
                    }
                    ```

                    with

                    `logging_fn = {"perplexity": lambda **kw: jnp.exp(kw["loss"])}`

                means the logging callable will receive:
                    ```
                    perplexity(loss=loss, nll=nll)
                    ```

                Example loss functions:
                    ```python
                    def loss_fn(model, batch):
                        logits = model(batch["input_ids"])
                        loss = cross_entropy_loss(logits, batch["labels"])
                        return loss
                    ```
                    ```
                    def loss_fn(model, batch):
                        logits = model(batch["input_ids"])
                        loss = cross_entropy_loss(logits, batch["labels"])
                        return {
                            "loss": loss,
                            "per_token_loss": loss,
                        }
                    ```

            train_dataset:
                Training dataset or dataset source accepted by `process_dataset`.

            eval_dataset:
                Optional evaluation dataset or dataset source accepted by
                `process_dataset`.

            config:
                Trainer configuration. If `None`, a default `TrainerConfig`
                is created.
        """
        
        self.loss_fn = loss_fn
        self.config = config = config or TrainerConfig()

        self.train_dataset, self._num_train_examples = process_dataset(
            train_dataset,
            config=config,
        )

        self.eval_dataset, self._num_eval_examples = None, None
        if eval_dataset is not None:
            self.eval_dataset, self._num_eval_examples = process_dataset(
                eval_dataset,
                config=config,
                is_eval=True
            )

        self.model, self.config = process_model(model, config)
        if config.remat:
            self.model = apply_remat(self.model)

    def train(self):
        """Run the full training loop.

        Handles:
            - Optimizer and LR schedule construction from config.
            - Gradient accumulation over micro-batches.
            - Multi-backend logging (stdout, TensorBoard, W&B, JSON).
            - Custom metric functions via trconfig.logging_fn.
            - Orbax CheckpointManager with automatic rotation.
            - Early stop at max_steps if set.
        """
        cfg = self.config

        # ── estimate total steps ──
        total_steps = cfg.max_steps
        if total_steps <= 0:
            if self._num_train_examples is None:
                raise ValueError("`max_steps` must be set when using a streaming dataset")
            per_device_batch = cfg.per_device_batch_size
            steps_per_epoch = self._num_train_examples // (per_device_batch * cfg.gradient_accumulation_steps)
            total_steps = steps_per_epoch * cfg.num_epochs

        if self.eval_dataset is not None:
            if cfg.eval_max_steps <= 0:
                if self._num_eval_examples is None:
                    raise ValueError("`eval_max_steps` must be set when using a eval streaming dataset")
                
                eval_batch_size = cfg.eval_per_device_batch_size or cfg.per_device_batch_size
                eval_total_steps = self._num_eval_examples // eval_batch_size
            else:
                eval_total_steps = cfg.eval_max_steps

        # ── build optimizer ──
        opt, schedule = (
            cfg.build_optimizer_fn and cfg.build_optimizer_fn(cfg, total_steps)
        ) or build_optimizer(cfg, total_steps)
        optimizer= nnx.Optimizer(self.model, opt, wrt=nnx.Param)

        # ── checkpoint manager (with built-in rotation) ──
        output_dir = Path(cfg.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt_options = ocp.CheckpointManagerOptions(
            max_to_keep=cfg.save_total_limit,
            create=True,
        )
        ckpt_mgr = ocp.CheckpointManager(
            output_dir,
            options=ckpt_options,
        )

        # ── logger ──
        logger = Logger(cfg.log_to, cfg.output_dir, cfg.run_name)
        model_name = type(self.model).__name__
        training_metric = TrainingMetric(model_name, self.config)

        # ── detect GaLore for projector updates ──
        galore_idx, galore_st = find_galore_state(optimizer.opt_state)
        is_galore = galore_st is not None
        galore_gap = cfg.optimizer_kwargs.get("galore_update_proj_gap", 200) if is_galore else 0
        galore_r = cfg.optimizer_kwargs.get("galore_r", 128) if is_galore else 0

        # ── training loop ──
        global_step = 0
        micro_step = 0
        accum_loss = 0.0
        accum_grads = None
        log_loss = 0.0
        t_start = time.time()

        sample_batch = next(iter(self.train_dataset))
        sample_batch = cfg.processing_train_dataset_fn(sample_batch) \
            if cfg.processing_train_dataset_fn is not None else sample_batch
        
        compute_loss_and_grads_fn = compute_loss_and_grads
        if cfg.jit_loss_fn:
            try:
                compute_loss_and_grads_fn = (
                    nnx.jit(
                        compute_loss_and_grads, 
                        static_argnames="loss_fn"
                    )
                    .lower(self.model, self.loss_fn, sample_batch)
                    .compile()
                )
            except TypeError as e:
                print(e)
                print()
                print(
                    "You may have values in your dataset batch that JAX cannot trace under `jit`.\n"
                    "This usually happens when the batch passed to `loss_fn` contains unsupported\n"
                    "types such as Python objects, strings, or NumPy arrays with incompatible dtypes,\n"
                    "instead of JAX-friendly array values.\n"
                    "\n"
                    "To fix this, you can:\n"
                    "  1. Preprocess your dataset so every field passed to `loss_fn` is JAX-compatible.\n"
                    "  2. Pass a preprocessing function to `preprocessing_dataset_fn` in `TrainerConfig`\n"
                    "     to transform dataset samples before they are passed to `loss_fn`.\n"
                    "  3. Set `jit_loss_fn=False` in `TrainerConfig` to disable JIT for the loss function."
                )
                sys.exit(0)

        model_state = nnx.state(self.model)
        trainable_state = nnx.state(self.model, nnx.Param)
        total_params = sum(
            leaf.size for leaf in jax.tree.leaves(model_state) if hasattr(leaf, "size")
        )
        trainable_params = sum(
            leaf.size for leaf in jax.tree.leaves(trainable_state) if hasattr(leaf, "size")
        )

        print_training_banner(
            model_name=model_name,
            total_params=total_params,
            trainable_params=trainable_params,
            **_get_training_banner_args(cfg, self.model, total_steps),
        )

        def _eval_logging():
            eval_metrics = {
                "eval_step": global_step,
                "loss": 0,
            }

            if cfg.logging_aux:
                for k, v in aux.items():
                    if hasattr(v, "item") and jnp.isscalar(v):
                        metrics[k] = 0
                    elif isinstance(v, (int, float)):
                        metrics[k] = 0
                        
            if cfg.logging_fn:
                for name, fn in cfg.logging_fn.items():
                    eval_metrics[name] = 0

            for i, eval_batch in tqdm(enumerate(self.eval_dataset), desc="eval", total=eval_total_steps):
                # ── pre-process batch dataset ──
                eval_batch = (
                    cfg.processing_eval_dataset_fn and \
                        cfg.processing_eval_dataset_fn(eval_batch)
                ) or eval_batch

                # ── forward + backward (micro-batch) ──
                loss, aux, grads = (
                    cfg.jit_loss_fn and compute_loss_and_grads_fn(self.model, batch)
                ) or compute_loss_and_grads_fn(self.model, self.loss_fn, batch)
                
                eval_metrics["loss"] += (hasattr(loss, "item") and loss.item()) or loss

                if cfg.logging_aux:
                    for k, v in aux.items():
                        if hasattr(v, "item") and jnp.isscalar(v):
                            metrics[k] += v.item()
                        elif isinstance(v, (int, float)):
                            metrics[k] += v

                if cfg.logging_fn:
                    for name, fn in cfg.logging_fn.items():
                        eval_metrics[name] += float(fn(**aux))

                if cfg.eval_max_steps > 0 and (i + 1) > eval_total_steps:
                    break

            eval_metrics["loss"] /= (i + 1)
            if cfg.logging_fn:
                for name, fn in cfg.logging_fn.items():
                    eval_metrics[name] /= (i + 1)

            logger.log(eval_metrics, step=global_step, is_eval=True)
            training_metric.update_training_metrics(global_step, eval_metrics, is_eval=True)

        with tqdm(total=total_steps, position=0, leave=True) as pbar:
            for epoch in range(cfg.num_epochs):
                for batch in self.train_dataset:
                    
                    # ── pre-process batch dataset ──
                    batch = (
                        cfg.processing_train_dataset_fn \
                            and cfg.processing_train_dataset_fn(batch)
                    ) or batch

                    # ── forward + backward (micro-batch) ──
                    loss, aux, grads = (
                        cfg.jit_loss_fn \
                            and compute_loss_and_grads_fn(self.model, batch)
                    ) or compute_loss_and_grads_fn(self.model, self.loss_fn, batch)
                    micro_step += 1

                    # ── accumulate gradients ──
                    if accum_grads is None:
                        accum_grads = grads
                    else:
                        accum_grads = jax.tree.map(jnp.add, accum_grads, grads)

                    accum_loss = accum_loss + loss

                    pbar.set_description(f"step {global_step + 1}/{total_steps}")
                    pbar.update(1)

                    # ── update when accumulation is complete ──
                    if micro_step % cfg.gradient_accumulation_steps == 0:
                        # average gradients
                        if cfg.gradient_accumulation_steps > 1:
                            accum_grads = jax.tree.map(
                                lambda g: g / cfg.gradient_accumulation_steps, accum_grads
                            )

                        # ── GaLore projector update (outside JIT) ──
                        if is_galore and global_step % galore_gap == 0:
                            params = nnx.state(self.model, nnx.Param)
                            cur_gstate = galore_st if galore_idx is None else optimizer.opt_state[galore_idx]
                            new_gstate = update_galore_projectors(cur_gstate, accum_grads, params, r=galore_r)
                            optimizer.opt_state = set_galore_state(optimizer.opt_state, galore_idx, new_gstate)
                            galore_st = new_gstate

                        def value_fn(model, loss_fn, batch):
                            if cfg.jit_loss_fn:
                                return compute_loss_and_grads_fn(model, batch)[0]
                            return compute_loss_and_grads_fn(model, loss_fn, batch)[0]
                        
                        def grad_fn(model, loss_fn, batch): 
                            if cfg.jit_loss_fn:
                                return compute_loss_and_grads_fn(model, batch)[2]
                            return compute_loss_and_grads_fn(model, loss_fn, batch)[2]
                        
                        def _compute_loss_and_grads_fn(model, loss_fn, batch):
                            if cfg.jit_loss_fn:
                                return compute_loss_and_grads_fn(model, batch)
                            return compute_loss_and_grads_fn(model, loss_fn, batch)

                        extra_args = (
                            cfg.return_optimizer_extra_args_fn \
                                and cfg.return_optimizer_extra_args_fn(
                                    loss, accum_grads, value_fn, 
                                    grad_fn, _compute_loss_and_grads_fn
                                )
                        ) or {
                            "value": jnp.array(loss),
                            "grad": accum_grads,
                            "value_fn": value_fn,
                            "grad_fn": grad_fn
                        }

                        optimizer.update(self.model, accum_grads, **extra_args)
                        grad_norm = optax.global_norm(accum_grads)
                        accum_grads = None
                        global_step += 1
                        avg_micro_loss = float(accum_loss) / cfg.gradient_accumulation_steps
                        log_loss += avg_micro_loss
                        accum_loss = 0.0

                        # ── logging ──
                        if global_step % cfg.logging_steps == 0:
                            elapsed = time.time() - t_start
                            avg_loss = log_loss / cfg.logging_steps

                            # base metrics + anything from loss_fn's dict return
                            metrics = {
                                "step": global_step,
                                "loss": avg_loss,
                                "learning_rate": schedule(global_step).item(),
                                "epoch": epoch,
                                "grad_norm": grad_norm.item(),
                                "steps_per_sec": cfg.logging_steps / elapsed,
                            }

                            if cfg.logging_aux:
                                for k, v in aux.items():
                                    if hasattr(v, "item") and jnp.isscalar(v):
                                        metrics[k] = v.item()
                                    elif isinstance(v, (int, float)):
                                        metrics[k] = v

                            # custom metric functions — receive full aux dict
                            if cfg.logging_fn:
                                for name, fn in cfg.logging_fn.items():
                                    result = fn(**aux)
                                    if hasattr(result, "item") and jnp.isscalar(result):
                                        metrics[name] = result.item()
                                    elif isinstance(result, (int, float)):
                                        metrics[name] = result

                            logger.log(metrics, step=global_step)
                            training_metric.update_training_metrics(global_step, metrics)
                            log_loss = 0.0
                            t_start = time.time()

                        # ── checkpointing (orbax handles rotation via max_to_keep) ──
                        if cfg.save_steps and global_step % cfg.save_steps == 0:
                            _, state = nnx.split(self.model)
                            ckpt_mgr.save(global_step, args=ocp.args.StandardSave(state))
                            ckpt_mgr.wait_until_finished()
                            training_metric.save_training_metrics()
                            tqdm.write(f"saved checkpoint → step {global_step}")

                        if self.eval_dataset is not None and cfg.eval_strategy == "steps" and global_step % cfg.eval_steps == 0:
                            _eval_logging()

                        # max steps reached
                        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                            break

                if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                    break

            if self.eval_dataset is not None \
                and cfg.eval_strategy == "epochs" \
                    and ((epoch + 1) % cfg.eval_epochs == 0 if cfg.eval_epochs is not None else True):
                _eval_logging()
                
            pbar.close()
        # ── save final ──
        _, state = nnx.split(self.model)
        ckpt_mgr.save(global_step, args=ocp.args.StandardSave(state))
        ckpt_mgr.wait_until_finished()
        training_metric.save_training_metrics()
        logger.close()
        ckpt_mgr.close()
        print(f"training complete — {global_step} steps | saved → {output_dir}")



# class SFTTrainer(Trainer):

#     """
#     # Experimental 
#     **use Trainer if face any problems**

#     Supervised Fine-Tuning Trainer.
#     Automates dataset tokenization, padding, and truncation.

#     """
#     def __init__(
#         self,
#         model,
#         dataset,
#         processor,
#         trconfig: Optional[SFTConfig] = None,
#         loss_fn: Optional[Callable] = None,
#     ):
#         trconfig = trconfig or SFTConfig()
        
#         if processor is None:
#             raise ValueError("SFTTrainer requires a `processor` (tokenizer) for dataset formatting.")
            
#         self.processor = processor
#         self.sft_config = trconfig
        
#         # Default causal LM loss if none explicitly provided
#         loss_fn = loss_fn or causal_lm_loss
        
#         # Auto-inject or wrap the DatasetConfig preprocessing function
#         original_preprocessing = dsconfig.preprocessing_fn
        
#         def sft_tokenizer_map(example):
#             # 1. Formatting
#             if self.sft_config.formatting_func is not None:
#                 text = self.sft_config.formatting_func(example)
#             else:
#                 text = example.get(self.sft_config.dataset_text_field, "")
#                 if not text and isinstance(example, dict):
#                     # fallback if "text" field is missing but dict has one key
#                     keys = list(example.keys())
#                     if len(keys) == 1:
#                         text = example[keys[0]]
            
#             # Allow the user's preprocessing to mutate the text/example first if they want
#             if original_preprocessing is not None:
#                 text = original_preprocessing(text)
                
#             # 2. Tokenize, Truncate, Pad
#             # We must enforce fixed lengths for JAX compiled loops
#             encoded = self.processor(
#                 text,
#                 padding="max_length",
#                 truncation=True,
#                 max_length=self.sft_config.max_seq_len,
#                 return_tensors="np" # JAX deals with numpy arrays before converting to jnp
#             )
            
#             # Strip the batch dimension [1, seq_len] -> [seq_len] that HF tokenizers add
#             return {
#                 "input_ids": encoded["input_ids"][0],
#                 "attention_mask": encoded["attention_mask"][0]
#             }
            
#         dsconfig.preprocessing_fn = sft_tokenizer_map
        
#         super().__init__(
#             model=model,
#             dataset=dataset,
#             loss_fn=loss_fn,
#             processor=processor,
#             trconfig=trconfig,
#         )


# class GRPOTrainer(Trainer):
#     """
#     # Experimental 
#     **use Trainer if face any problems**
    
#     Group Relative Policy Optimization Trainer.
#     (Placeholder for GRPO-specific group sampling and reward formulation)
#     """
#     def __init__(
#         self,
#         model,
#         ref_model, # Required for GRPO KL penalty
#         reward_funcs: list[Callable], # E.g., rule-based metrics
#         dataset,
#         processor=None,
#         trconfig: Optional[GRPOConfig] = None,
#         dsconfig=None,
#         loss_fn: Optional[Callable] = None,
#     ):
#         trconfig = trconfig or GRPOConfig()
#         self.ref_model = ref_model
#         self.reward_funcs = reward_funcs
        
#         # Default to standard GRPO loss if none provided
#         loss_fn = loss_fn or grpo_loss_fn
        
#         super().__init__(
#             model=model,
#             dataset=dataset,
#             loss_fn=loss_fn,
#             processor=processor,
#             trconfig=trconfig,
#             dsconfig=dsconfig
#         )

#     def train(self):
#         """
#         Run the GRPO training loop with generation rollouts.
#         For each batch (prompt):
#         1. Generate G completions using the current policy (model.generate)
#         2. Evaluate string completions through reward_funcs
#         3. Compute group-relative advantages
#         4. Capture static log probabilities (old_lp, ref_lp)
#         5. Map a synthetic batch dictionary and run optimization steps
#         """
#         import time
#         from pathlib import Path
#         import orbax.checkpoint as ocp
#         import numpy as np
        
#         from .trainer import build_optimizer, Logger

#         cfg = self.trconfig
#         total_steps = cfg.max_steps
#         if total_steps <= 0:
#             raise ValueError("GRPOTrainer requires an explicit max_steps configuration.")

#         # Optimizer
#         opt = build_optimizer(cfg, total_steps)
#         optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)

#         # Checkpoints
#         output_dir = Path(cfg.output_dir).resolve()
#         output_dir.mkdir(parents=True, exist_ok=True)
#         ckpt_mgr = ocp.CheckpointManager(
#             output_dir / "checkpoints",
#             options=ocp.CheckpointManagerOptions(max_to_keep=cfg.save_total_limit, create=True),
#         )

#         logger = Logger(cfg.log_to, cfg.output_dir, cfg.run_name)

#         # Retrieve parameters
#         G = cfg.num_generations
#         max_prompt_len = cfg.max_prompt_length
#         max_seq_len = cfg.max_seq_len
#         max_new_tokens = max_seq_len - max_prompt_len

#         global_step = 0
#         t_start = time.time()
        
#         from ..models.infer import _forward_step

#         def _micro_forward_log_probs(mdl, ids, mask):
#             """Forward function to get tokens log-probs without triggering large JIT overheads."""
#             pos = jnp.maximum(jnp.cumsum(mask.astype(jnp.int32), axis=-1) - 1, 0)
#             logits = mdl(ids, mask, pos).logits
#             return token_log_probs(logits, ids)

#         for epoch in range(cfg.num_epochs):
#             for batch_prompts in self.dataset:
#                 # `batch_prompts` should dictate purely the prompt input ids and attention mask
#                 prompt_ids = batch_prompts["input_ids"] # (B, P)
#                 prompt_mask = batch_prompts["attention_mask"] # (B, P)
                
#                 # Assume batch size 1 for rollouts for simplicity of generation loop mappings
#                 prompt_ids = prompt_ids[0:1]
#                 prompt_mask = prompt_mask[0:1]
#                 prompt_len = int(prompt_mask.sum())

#                 # 1. Generate G Completions
#                 rep_ids = jnp.repeat(prompt_ids, G, axis=0)
#                 rep_mask = jnp.repeat(prompt_mask, G, axis=0)
                
#                 # Use model properties to perform live generation
#                 full_ids = self.model.generate(
#                     rep_ids, rep_mask,
#                     key=jax.random.key(global_step),
#                     max_new_tokens=max_new_tokens,
#                 )
                
#                 # Turn off cache usage for forward loops
#                 for _, m in nnx.iter_modules(self.model):
#                     if hasattr(m, "use_cache"):
#                         m.use_cache = False

#                 # 2. Extract texts and compute reward
#                 completions = [
#                     self.processor.decode(full_ids[j, prompt_len:], skip_special_tokens=True)
#                     for j in range(G)
#                 ]
                
#                 rewards = np.zeros(G, dtype=np.float32)
#                 for i, comp in enumerate(completions):
#                     # Sum all user defined metrics
#                     r = sum(func(comp) for func in self.reward_funcs)
#                     rewards[i] = r
                
#                 # 3. Advantages [(r - μ) / σ]
#                 adv_std = rewards.std() + 1e-8
#                 advantages = jnp.array((rewards - rewards.mean()) / adv_std)
                
#                 # 4. Freeze log probs for optimization
#                 full_mask = jnp.ones_like(full_ids, dtype=jnp.int32)
                
#                 # Current Policy
#                 old_lp = jax.lax.stop_gradient(_micro_forward_log_probs(self.model, full_ids, full_mask))
                
#                 # Reference Policy
#                 if self.ref_model is not None:
#                     for _, m in nnx.iter_modules(self.ref_model):
#                         if hasattr(m, "use_cache"):
#                             m.use_cache = False
#                     ref_lp = jax.lax.stop_gradient(_micro_forward_log_probs(self.ref_model, full_ids, full_mask))
#                 else:
#                     ref_lp = old_lp

#                 # 5. Inner Optimization Loop (PPO style mu_epochs)
#                 synthetic_batch = {
#                     "input_ids": full_ids,
#                     "attention_mask": full_mask,
#                     "advantages": advantages,
#                     "old_lp": old_lp,
#                     "ref_lp": ref_lp,
#                     "prompt_len": prompt_len,
#                     "clip_eps": cfg.clip_eps,
#                     "kl_coeff": cfg.beta,
#                     "entropy_coeff": cfg.entropy_coeff,
#                 }
                
#                 for _ in range(cfg.mu_epochs):
#                     loss, grads = nnx.value_and_grad(self.loss_fn, argnums=nnx.DiffState(0, nnx.Param))(
#                         self.model, synthetic_batch
#                     )
#                     optimizer.update(self.model, grads)

#                 global_step += 1

#                 # 6. Logging
#                 if global_step % cfg.logging_steps == 0:
#                     metrics = {
#                         "loss": float(loss),
#                         "reward_mean": float(rewards.mean()),
#                         "reward_std": float(rewards.std()),
#                         "learning_rate": float(opt.schedule(global_step) if hasattr(opt, "schedule") else cfg.learning_rate),
#                         "global_step": global_step,
#                         "epoch": epoch,
#                     }
#                     if cfg.logging_fn is not None:
#                         for k, v in cfg.logging_fn.items():
#                             metrics[k] = float(v(**metrics))
#                     logger.log(metrics, step=global_step)

#                 # 7. Checkpointing
#                 if global_step % cfg.save_steps == 0:
#                     model_state = nnx.state(self.model)
#                     ckpt_mgr.save(global_step, args=ocp.args.StandardSave(model_state))

#                 if global_step >= total_steps:
#                     break
#             if global_step >= total_steps:
#                 break

#         # Final save
#         model_state = nnx.state(self.model)
#         ckpt_mgr.save(global_step, args=ocp.args.StandardSave(model_state), force=True)
#         ckpt_mgr.wait_until_finished()
#         logger.close()


# class DSFTTrainer(Trainer):
#     """
#     # Experimental 
#     **use Trainer if face any problems**
    
#     Diffusion Supervised Fine-Tuning Trainer.
#     Automates image transformations and the discrete noise scheduler mapping.
#     """
#     def __init__(
#         self,
#         model,
#         dataset,
#         noise_scheduler,
#         processor=None, # optional text tokenizer/image processor
#         trconfig: Optional[DSFTConfig] = None,
#         dsconfig: Optional[DatasetConfig] = None,
#         loss_fn: Optional[Callable] = None,
#     ):
#         trconfig = trconfig or DSFTConfig()
#         dsconfig = dsconfig or DatasetConfig()
        
#         self.noise_scheduler = noise_scheduler
#         self.processor = processor
#         self.dsft_config = trconfig
        
#         loss_fn = loss_fn or diffusion_loss
#         original_preprocessing = dsconfig.preprocessing_fn
        
#         def diffusion_map(example):
#              # 1. Apply any custom formatting or raw value extraction
#             if self.dsft_config.formatting_func is not None:
#                 features = self.dsft_config.formatting_func(example)
#             else:
#                 features = {
#                     "image": example[self.dsft_config.dataset_image_field],
#                 }
#                 if self.dsft_config.dataset_text_field in example:
#                      features["text"] = example[self.dsft_config.dataset_text_field]

#             if original_preprocessing is not None:
#                 features = original_preprocessing(features)
                
#             # 2. Extract cleanly shaped pixel values and tokenize conditionings
#             # Images are theoretically converted to [C, H, W] arrays here via external transforms
#             pixel_values = jnp.array(features["image"]) 
            
#             out = {"clean_images": pixel_values}
            
#             if "text" in features and self.processor is not None:
#                 encoded = self.processor(
#                     features["text"],
#                     padding="max_length",
#                     truncation=True,
#                     max_length=77, # standard strict CLIP horizon
#                     return_tensors="np"
#                 )
#                 out["conditioning"] = encoded["input_ids"][0]
                
#             # IMPORTANT: The dynamic forward noising processes (sampling `t` and adding `noise()`)
#             # cannot be done inside preprocessing because Grain Datasets execute map_fns purely statically
#             # on the host prior to device sharding. To get distinct random shapes and timesteps per batch,
#             # we must process the final raw arrays.
#             # INSTEAD: The Trainer automatically requires a hook mapping if you don't noise directly in loss.
#             # To stick to best practices, users should wrap `diffusion_loss` to sample `t` JIT-side if needed, 
#             # or we do it heuristically. Here we leave the batch pure, and assume the loss function handles noising!
            
#             return out
            
#         dsconfig.preprocessing_fn = diffusion_map
        
#         super().__init__(
#             model=model,
#             dataset=dataset,
#             loss_fn=loss_fn,
#             processor=processor,
#             trconfig=trconfig,
#             dsconfig=dsconfig
#         )


# class DPOTrainer(Trainer):
#     """
#     # Experimental 
#     **use Trainer if face any problems**
    
#     Direct Preference Optimization Trainer.
#     (Placeholder for DPO-specific dataset formatting and pairwise loss computation)
#     """
#     def __init__(
#         self,
#         model,
#         ref_model, # Required for DPO KL divergence penalty
#         dataset,
#         processor=None,
#         trconfig: Optional[DPOConfig] = None,
#         dsconfig=None,
#         loss_fn: Optional[Callable] = None,
#     ):
#         import numpy as np

#         trconfig = trconfig or DPOConfig()
#         self.ref_model = ref_model
        
#         loss_fn = loss_fn or dpo_loss_fn
        
#         # Determine strict integer mapping for JIT loss branching
#         loss_types = {"sigmoid": 0, "hinge": 1, "ipo": 2, "kto_pair": 3}
#         loss_type_id = loss_types.get(trconfig.loss_type.lower(), 0)

#         # Build automated processor mapping if tokenizer is provided
#         if processor is not None and dsconfig is None:
#             def dpo_preprocess(example):
#                 """
#                 Map a single HF dictionary row containing 'prompt', 'chosen', 'rejected'
#                 into 'chosen_input_ids', 'rejected_input_ids', etc.
#                 """
#                 prompt = example[trconfig.dataset_prompt_field]
#                 chosen = example[trconfig.dataset_chosen_field]
#                 rejected = example[trconfig.dataset_rejected_field]
                
#                 chosen_text = prompt + chosen
#                 rejected_text = prompt + rejected
                
#                 chosen_enc = processor(
#                     chosen_text,
#                     padding="max_length",
#                     truncation=True,
#                     max_length=trconfig.max_seq_len,
#                     return_tensors="np"
#                 )
                
#                 rejected_enc = processor(
#                     rejected_text,
#                     padding="max_length",
#                     truncation=True,
#                     max_length=trconfig.max_seq_len,
#                     return_tensors="np"
#                 )
                
#                 # Processor returns shape (1, S) or (S,), we ensure 1D shape (S,)
#                 c_ids = chosen_enc["input_ids"].squeeze()
#                 c_mask = chosen_enc["attention_mask"].squeeze()
#                 r_ids = rejected_enc["input_ids"].squeeze()
#                 r_mask = rejected_enc["attention_mask"].squeeze()
                
#                 return {
#                     "chosen_input_ids": c_ids,
#                     "chosen_attention_mask": c_mask,
#                     "rejected_input_ids": r_ids,
#                     "rejected_attention_mask": r_mask,
#                     "prompt_len": np.array(trconfig.max_prompt_length, dtype=np.int32),
#                     "beta": np.array(trconfig.beta, dtype=np.float32),
#                     "loss_type": np.array(loss_type_id, dtype=np.int32),
#                     "label_smoothing": np.array(trconfig.label_smoothing, dtype=np.float32),
#                 }
                
#             dsconfig = DatasetConfig(preprocessing_fn=dpo_preprocess)

#         super().__init__(
#             model=model,
#             dataset=dataset,
#             loss_fn=loss_fn,
#             processor=processor,
#             trconfig=trconfig,
#             dsconfig=dsconfig
#         )

#     def train(self):
#         """
#         Run the DPO training loop.
#         Overrides the standard Trainer.train() to inject `ref_chosen_lp` 
#         and `ref_rejected_lp` from the frozen reference model into each batch natively, 
#         saving the loss function from needing to hold its own stateful references.
#         """
#         import time
#         from pathlib import Path
#         import orbax.checkpoint as ocp
#         from flax import nnx
#         import jax
#         import jax.numpy as jnp
        
#         from ..trainer.trainer import build_optimizer, Logger, compute_loss_and_grads

#         cfg = self.trconfig
#         total_steps = cfg.max_steps
#         if total_steps <= 0:
#             if self._num_examples is None:
#                 raise ValueError("max_steps must be set when using a streaming dataset")
#             per_device_batch = cfg.per_device_batch_size
#             steps_per_epoch = self._num_examples // (per_device_batch * cfg.gradient_accumulation_steps)
#             total_steps = steps_per_epoch * cfg.num_epochs

#         opt = build_optimizer(cfg, total_steps)
#         optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)

#         output_dir = Path(cfg.output_dir).resolve()
#         output_dir.mkdir(parents=True, exist_ok=True)
#         ckpt_mgr = ocp.CheckpointManager(
#             output_dir / "checkpoints",
#             options=ocp.CheckpointManagerOptions(max_to_keep=cfg.save_total_limit, create=True),
#         )

#         logger = Logger(cfg.log_to, cfg.output_dir, cfg.run_name)
        
#         # Micro JIT forward function for the reference model to prevent memory duplication in training graph
#         @jax.jit
#         def _get_ref_logps(ref_model, ch_ids, ch_mask, rej_ids, rej_mask, prompt_len):
#             # 1. Chosen
#             ch_pos = jnp.maximum(jnp.cumsum(ch_mask.astype(jnp.int32), axis=-1) - 1, 0)
#             ch_logits = ref_model(ch_ids, ch_mask, ch_pos).logits
#             ch_lp = token_log_probs(ch_logits, ch_ids)
#             ch_comp_mask = (jnp.arange(ch_lp.shape[1]) >= (prompt_len - 1)) & (ch_mask[:, 1:] > 0)
#             ch_logps = (ch_lp * ch_comp_mask).sum(-1)
            
#             # 2. Rejected
#             rej_pos = jnp.maximum(jnp.cumsum(rej_mask.astype(jnp.int32), axis=-1) - 1, 0)
#             rej_logits = ref_model(rej_ids, rej_mask, rej_pos).logits
#             rej_lp = token_log_probs(rej_logits, rej_ids)
#             rej_comp_mask = (jnp.arange(rej_lp.shape[1]) >= (prompt_len - 1)) & (rej_mask[:, 1:] > 0)
#             rej_logps = (rej_lp * rej_comp_mask).sum(-1)
            
#             return ch_logps, rej_logps

#         global_step = 0
#         micro_step = 0
#         accum_loss = 0.0
#         accum_grads = None
#         log_loss = 0.0
        
#         for epoch in range(cfg.num_epochs):
#             for batch in self.dataset:
                
#                 # --- INJECT REFERENCE LOG PROBS ---
#                 if self.ref_model is not None:
#                     # Turn off KV cache to avoid state mutations
#                     for _, m in nnx.iter_modules(self.ref_model):
#                         if hasattr(m, "use_cache"):
#                             m.use_cache = False
                            
#                     ch_lp, rej_lp = jax.lax.stop_gradient(_get_ref_logps(
#                         self.ref_model,
#                         batch["chosen_input_ids"], batch["chosen_attention_mask"],
#                         batch["rejected_input_ids"], batch["rejected_attention_mask"],
#                         batch["prompt_len"][0] # Assuming uniform prompt length for batch mapping
#                     ))
#                 else:
#                     # Fallback (KL == 0 constraint absent)
#                     batch_size = batch["chosen_input_ids"].shape[0]
#                     ch_lp = jnp.zeros((batch_size,), dtype=jnp.float32)
#                     rej_lp = jnp.zeros((batch_size,), dtype=jnp.float32)
                    
#                 batch["ref_chosen_lp"] = ch_lp
#                 batch["ref_rejected_lp"] = rej_lp
#                 # ----------------------------------

#                 # Standard Micro-Batch Loop
#                 loss, aux, grads = compute_loss_and_grads(self.model, self.loss_fn, batch)
#                 micro_step += 1

#                 if accum_grads is None:
#                     accum_grads = grads
#                 else:
#                     accum_grads = jax.tree.map(lambda a, g: a + g, accum_grads, grads)
                
#                 accum_loss += loss
#                 log_loss += float(loss)

#                 # Gradient optimization step
#                 if micro_step == cfg.gradient_accumulation_steps:
#                     accum_grads = jax.tree.map(lambda g: g / cfg.gradient_accumulation_steps, accum_grads)
#                     optimizer.update(self.model, accum_grads)
#                     global_step += 1
                    
#                     if global_step % cfg.logging_steps == 0:
#                         avg_loss = log_loss / (cfg.logging_steps * cfg.gradient_accumulation_steps)
#                         metrics = {
#                             "loss": avg_loss,
#                             "learning_rate": float(opt.schedule(global_step) if hasattr(opt, "schedule") else cfg.learning_rate),
#                             "global_step": global_step,
#                             "epoch": epoch,
#                         }
#                         # Add any dict outputs from dpo_loss_fn (if aux is returning one)
#                         if isinstance(aux, dict):
#                             for k, v in aux.items():
#                                 if k != "loss":
#                                     metrics[k] = float(v)
                                    
#                         if cfg.logging_fn is not None:
#                             for k, v in cfg.logging_fn.items():
#                                 metrics[k] = float(v(**metrics))
                                
#                         logger.log(metrics, step=global_step)
#                         log_loss = 0.0

#                     if global_step % cfg.save_steps == 0:
#                         model_state = nnx.state(self.model)
#                         ckpt_mgr.save(global_step, args=ocp.args.StandardSave(model_state))

#                     micro_step = 0
#                     accum_loss = 0.0
#                     accum_grads = None

#                     if global_step >= total_steps:
#                         break
#             if global_step >= total_steps:
#                 break

#         model_state = nnx.state(self.model)
#         ckpt_mgr.save(global_step, args=ocp.args.StandardSave(model_state), force=True)
#         ckpt_mgr.wait_until_finished()
#         logger.close()
