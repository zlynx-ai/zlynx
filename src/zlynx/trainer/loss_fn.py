
import jax, jax.numpy as jnp
from flax import nnx


def causal_lm_loss(model, batch):
    """
    Standard autoregressive causal language modeling loss.
    Assumes `batch` has "input_ids" and (optionally) "labels" and "attention_mask".
    Delegates calculation natively to the model outputs.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)
    labels = batch.get("labels", input_ids)
    
    # Forward pass: calculates shifted cross-entropy internally
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    
    return outputs.loss



# ─────────────────────────────────────────────────────────────
# Training step (JIT-compiled)
# ─────────────────────────────────────────────────────────────

# @nnx.jit(static_argnames="loss_fn")
def compute_loss_and_grads(model, loss_fn, batch):
    """Compute loss, aux data, and gradients for a single micro-batch.

    Args:
        model: An nnx.Module to differentiate w.r.t.
        loss_fn: Callable (model, batch) → scalar loss OR dict with "loss" key.
            If a dict is returned, "loss" is used for backprop and the full
            dict is forwarded to logging_fn as **kwargs.
        batch: A single micro-batch (dict, array, etc.).

    Returns:
        Tuple of (loss, aux, grads) where:
            loss: scalar loss value
            aux: dict of all returned values (always contains "loss")
            grads: pytree matching the model's parameters
    """
    def wrapped(model, batch):
        result = loss_fn(model, batch)
        if isinstance(result, dict):
            return result["loss"], result
        return result, {"loss": result}

    (loss, aux), grads = nnx.value_and_grad(
        wrapped, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
    )(model, batch)
    return loss, aux, grads

def token_log_probs(logits, labels):
    """Per-token log P(label | context).  (B, S-1)"""
    lp = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
    return jnp.take_along_axis(lp, labels[:, 1:, None], axis=-1).squeeze(-1)

def grpo_loss_fn(model, batch):
    """
    Standard Group Relative Policy Optimization loss.
    Expects `batch` to contain:
    - input_ids (B, S)
    - attention_mask (B, S)
    - advantages (B,)
    - old_lp (B, S-1)
    - ref_lp (B, S-1)
    - prompt_len (scalar int)
    - clip_eps (scalar float)
    - kl_coeff (scalar float)
    - entropy_coeff (scalar float)
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    advantages = batch["advantages"]
    old_lp = batch["old_lp"]
    ref_lp = batch["ref_lp"]
    
    prompt_len = batch["prompt_len"]
    clip_eps = batch.get("clip_eps", 0.2)
    kl_coeff = batch.get("kl_coeff", 0.04)
    entropy_coeff = batch.get("entropy_coeff", 0.01)

    # Forward pass on current policy
    pos = jnp.maximum(jnp.cumsum(attention_mask.astype(jnp.int32), axis=-1) - 1, 0)
    outputs = model(input_ids, attention_mask, pos)
    logits = outputs.logits
    cur_lp = token_log_probs(logits, input_ids)

    # Completion mask (only evaluate loss on generated tokens)
    seq_len = cur_lp.shape[1]
    comp_mask = (jnp.arange(seq_len) >= (prompt_len - 1)) & (attention_mask[:, 1:] > 0)

    # 1. Policy gradient
    ratio = jnp.exp(cur_lp - old_lp)
    adv = advantages[:, None]
    pg = jnp.minimum(
        ratio * adv,
        jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv,
    )

    # 2. KL penalty
    log_r = ref_lp - cur_lp
    kl = jnp.exp(log_r) - log_r - 1

    # 3. Entropy bonus
    probs = jax.nn.softmax(logits[:, :-1, :], axis=-1)
    ent = -(probs * jax.nn.log_softmax(logits[:, :-1, :], axis=-1)).sum(-1)

    # Aggregate (TRL Default / DAPO formulation)
    # Instead of averaging the per-sequence averages (1/G * sum(1/|o_i| * sum(...))),
    # TRL divides the global sum of token losses by the global sum of valid tokens.
    total_tokens = comp_mask.sum().clip(min=1)
    
    pg_loss = -((pg * comp_mask).sum() / total_tokens)
    kl_loss = ((kl * comp_mask).sum() / total_tokens)
    ent_loss = -((ent * comp_mask).sum() / total_tokens) # Maximise entropy

    loss = pg_loss + kl_coeff * kl_loss + entropy_coeff * ent_loss
    return loss


def diffusion_loss(model, batch):
    """
    Standard Diffusion Model Loss.
    Assumes `batch` contains pre-computed arrays or handles dynamic noising.
    """
    noisy_images = batch["noisy_images"]
    timesteps = batch["timesteps"]
    target_noise = batch["noise"]
    
    cond = batch.get("conditioning", None)
    
    if cond is not None:
        noise_pred = model(noisy_images, timesteps, cond)
    else:
        noise_pred = model(noisy_images, timesteps)
        
    # Standard MSE loss
    loss = jnp.mean((noise_pred - target_noise) ** 2)
    return loss


def dpo_loss_fn(model, batch):
    """
    Direct Preference Optimization loss.
    Expects `batch` to contain:
    - chosen_input_ids (B, S)
    - chosen_attention_mask (B, S)
    - rejected_input_ids (B, S)
    - rejected_attention_mask (B, S)
    - prompt_len (scalar int)
    - ref_chosen_lp (B,) Total log prob of chosen completion under reference model
    - ref_rejected_lp (B,) Total log prob of rejected completion under reference model
    - beta (scalar float)
    - loss_type (string id)
    - label_smoothing (scalar float)
    """
    beta = batch.get("beta", 0.1)
    loss_type = batch.get("loss_type", 0) # 0: sigmoid, 1: hinge, 2: ipo, 3: kto_pair
    label_smoothing = batch.get("label_smoothing", 0.0)
    prompt_len = batch["prompt_len"]
    
    # 1. Forward Pass Chosen
    ch_ids = batch["chosen_input_ids"]
    ch_mask = batch["chosen_attention_mask"]
    ch_pos = jnp.maximum(jnp.cumsum(ch_mask.astype(jnp.int32), axis=-1) - 1, 0)
    
    ch_logits = model(ch_ids, ch_mask, ch_pos).logits
    ch_lp = token_log_probs(ch_logits, ch_ids)           # (B, S-1)
    
    # Ignore prompt tokens
    seq_len = ch_lp.shape[1]
    ch_comp_mask = (jnp.arange(seq_len) >= (prompt_len[:, None] - 1)) & (ch_mask[:, 1:] > 0)
    policy_chosen_logps = (ch_lp * ch_comp_mask).sum(-1) # (B,)
    
    # 2. Forward Pass Rejected
    rej_ids = batch["rejected_input_ids"]
    rej_mask = batch["rejected_attention_mask"]
    rej_pos = jnp.maximum(jnp.cumsum(rej_mask.astype(jnp.int32), axis=-1) - 1, 0)
    
    rej_logits = model(rej_ids, rej_mask, rej_pos).logits
    rej_lp = token_log_probs(rej_logits, rej_ids)        # (B, S-1)
    
    rej_comp_mask = (jnp.arange(seq_len) >= (prompt_len[:, None] - 1)) & (rej_mask[:, 1:] > 0)
    policy_rejected_logps = (rej_lp * rej_comp_mask).sum(-1) # (B,)
    
    # 3. Reference Model Log Probs (provided in batch to avoid dual-forward pass in loss_fn)
    ref_chosen_logps = batch["ref_chosen_lp"]
    ref_rejected_logps = batch["ref_rejected_lp"]
    
    # 4. Computed Implicit Rewards
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    
    # 5. Determine the mathematical loss type
    # For JIT compatibility, we use entirely explicit boolean masking for string equivalence
    is_sigmoid = loss_type == 0
    is_hinge = loss_type == 1
    is_ipo = loss_type == 2
    is_kto = loss_type == 3
    
    # Sigmoid (Standard DPO)
    loss_sigmoid = -jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing) - jax.nn.log_sigmoid(-beta * logits) * label_smoothing
    
    # Hinge Loss
    loss_hinge = jax.nn.relu(1 - beta * logits)
    
    # IPO Loss
    loss_ipo = (logits - 1/(2 * beta)) ** 2
    
    # KTO Pair Loss
    chosen_KL = policy_chosen_logps - ref_chosen_logps
    rejected_KL = policy_rejected_logps - ref_rejected_logps
    
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    
    loss_kto = 1 - jax.nn.sigmoid(beta * (chosen_logratios - rejected_KL)) + 1 - jax.nn.sigmoid(beta * (chosen_KL - rejected_logratios))
    
    # Combine (only 1 will be active due to JIT integer mapping)
    loss = (loss_sigmoid * is_sigmoid) + (loss_hinge * is_hinge) + (loss_ipo * is_ipo) + (loss_kto * is_kto)
    
    return loss.mean()
