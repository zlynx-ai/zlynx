from typing import NamedTuple, Any
import optax
import jax
import jax.numpy as jnp

from .trainer import TrainerConfig


OPTIMIZERS = {
    "adamw": optax.adamw,
    "adam": optax.adam,
    "sgd": optax.sgd,
    "lion": optax.lion,
}

SCHEDULERS = {
    "constant": lambda lr, total, warmup: optax.constant_schedule(lr),
    "linear": lambda lr, total, warmup: optax.linear_schedule(lr, 0.0, total),
    "cosine": lambda lr, total, warmup: optax.cosine_decay_schedule(lr, total),
    "warmup_cosine": lambda lr, total, warmup: optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup, decay_steps=total
    ),
}


class GaloreState(NamedTuple):
    inner_state: Any
    projector: Any
    step: jnp.ndarray

def galore_wrapper(inner_opt: optax.GradientTransformation, r: int = 128, update_proj_gap: int = 200, scale: float = 1.0):
    """
    Wraps an Optax optimizer to apply Gradient Low-Rank Projection (GaLore).
    This reduces the memory footprint of optimizer states (like Adam's moments)
    by projecting gradients into a low-rank subspace.

    The SVD-based projector update runs outside JIT via `update_galore_projectors()`.
    The optimizer's update_fn only does the cheap project → inner opt → project back.
    """
    def init_fn(params):
        def _get_lr_shape(p):
            if p.ndim < 2 or min(p.shape) <= r:
                return p
            m, n = p.shape
            return jnp.zeros((m, r) if m < n else (r, n), dtype=p.dtype)

        low_rank_params = jax.tree_util.tree_map(_get_lr_shape, params)
        inner_state = inner_opt.init(low_rank_params)

        def _init_proj(p):
            if p.ndim < 2 or min(p.shape) <= r:
                return None
            m, n = p.shape
            if m < n:
                return jnp.zeros((n, r), dtype=p.dtype)
            else:
                return jnp.zeros((m, r), dtype=p.dtype)

        projector = jax.tree_util.tree_map(_init_proj, params)
        return GaloreState(inner_state=inner_state, projector=projector, step=jnp.array(0, dtype=jnp.int32))

    def update_fn(updates, state, params=None):
        # Project gradients down using current projectors (no SVD here)
        def _project_down(update, param, proj):
            if proj is None:
                return update
            m, n = param.shape
            if m < n:
                return jnp.dot(update, proj)    # (m, r)
            else:
                return jnp.dot(proj.T, update)  # (r, n)

        low_rank_updates = jax.tree_util.tree_map(
            _project_down, updates, params, state.projector,
            is_leaf=lambda x: x is None
        )

        # Project params down for inner optimizer
        def _get_lr_params(orig_p, proj):
            if proj is None or orig_p is None: return orig_p
            m, n = orig_p.shape
            if m < n:
                return jnp.dot(orig_p, proj)
            else:
                return jnp.dot(proj.T, orig_p)

        lr_params = jax.tree_util.tree_map(
            _get_lr_params, params, state.projector,
            is_leaf=lambda x: x is None
        ) if params is not None else None

        # Inner optimizer step in low-rank space
        inner_updates_lr, new_inner_state = inner_opt.update(low_rank_updates, state.inner_state, lr_params)

        # Project updates back to full rank
        def _project_up(inner_u, orig_p, proj):
            if proj is None: return inner_u * scale
            m, n = orig_p.shape
            if m < n:
                return jnp.dot(inner_u, proj.T) * scale
            else:
                return jnp.dot(proj, inner_u) * scale

        final_updates = jax.tree_util.tree_map(
            _project_up, inner_updates_lr, params, state.projector,
            is_leaf=lambda x: x is None
        )

        return final_updates, GaloreState(
            inner_state=new_inner_state,
            projector=state.projector,
            step=state.step + 1,
        )

    return optax.GradientTransformation(init_fn, update_fn)


def update_galore_projectors(galore_state, grads, params, r: int = 128):
    """Update GaLore projectors via SVD. Call from Python every `update_proj_gap` steps.

    This runs outside JIT — SVD is computed eagerly on device, avoiding the
    catastrophic XLA compilation cost of tracing SVD for every parameter.

    Args:
        galore_state: The GaloreState from the optimizer.
        grads: The current gradient pytree (same structure as params).
        params: The current parameter pytree.
        r: Projection rank (must match the rank used in galore_wrapper).

    Returns:
        Updated GaloreState with new projectors.
    """
    # Flatten to raw arrays to avoid pytree structure mismatch between
    # nnx variable types (Param, OptVariable, etc.)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    param_leaves = jax.tree_util.tree_leaves(params)
    proj_flat, proj_treedef = jax.tree_util.tree_flatten(
        galore_state.projector, is_leaf=lambda x: x is None
    )

    new_proj_flat = []
    gi = 0  # grad/param index (only count leaves that have a projector)
    for proj in proj_flat:
        if proj is None:
            new_proj_flat.append(None)
            gi += 1
            continue
        grad = grad_leaves[gi]
        param = param_leaves[gi]
        gi += 1
        m, n = param.shape
        U, S, Vh = jnp.linalg.svd(grad.astype(jnp.float32), full_matrices=False)
        if m < n:
            new_proj_flat.append(Vh[:r, :].T.astype(grad.dtype))
        else:
            new_proj_flat.append(U[:, :r].astype(grad.dtype))

    new_projectors = proj_treedef.unflatten(new_proj_flat)

    return GaloreState(
        inner_state=galore_state.inner_state,
        projector=new_projectors,
        step=galore_state.step,
    )


def find_galore_state(opt_state):
    """Find GaloreState index in an optax chain. Returns (index, state) or (None, state)."""
    if isinstance(opt_state, GaloreState):
        return None, opt_state
    if isinstance(opt_state, tuple):
        for i, s in enumerate(opt_state):
            if isinstance(s, GaloreState):
                return i, s
    return None, None


def set_galore_state(opt_state, idx, new_gstate):
    """Replace GaloreState in an optax chain."""
    if idx is None:
        return new_gstate
    new_state = list(opt_state)
    new_state[idx] = new_gstate
    return tuple(new_state)


def build_optimizer(trconfig: "TrainerConfig", total_steps: int):
    """Build an optax optimizer chain from TrainerConfig."""
    warmup = trconfig.warmup_steps or int(trconfig.warmup_ratio * total_steps)

    schedule_fn = SCHEDULERS.get(trconfig.lr_scheduler, SCHEDULERS["cosine"])
    schedule = schedule_fn(trconfig.learning_rate, total_steps, warmup)

    opt_fn = OPTIMIZERS.get(trconfig.optimizer.replace("galore_", ""))
    if opt_fn is None:
        raise ValueError(f"Unknown optimizer: {trconfig.optimizer}. Available inner: {list(OPTIMIZERS.keys())}")

    inner_kwargs = {k: v for k, v in trconfig.optimizer_kwargs.items() if not k.startswith("galore_")}

    if "adamw" in trconfig.optimizer:
        opt = opt_fn(learning_rate=schedule, weight_decay=trconfig.weight_decay, **inner_kwargs)
    elif "sgd" in trconfig.optimizer:
        opt = opt_fn(learning_rate=schedule, **inner_kwargs)
    else:
        opt = opt_fn(learning_rate=schedule, **inner_kwargs)
        
    if trconfig.optimizer.startswith("galore_"):
        r = trconfig.optimizer_kwargs.get("galore_r", 128)
        update_gap = trconfig.optimizer_kwargs.get("galore_update_proj_gap", 200)
        scale = trconfig.optimizer_kwargs.get("galore_scale", 1.0)
        opt = galore_wrapper(opt, r=r, update_proj_gap=update_gap, scale=scale)

    if trconfig.max_grad_norm is not None:
        opt = optax.chain(optax.clip_by_global_norm(trconfig.max_grad_norm), opt)

    return opt