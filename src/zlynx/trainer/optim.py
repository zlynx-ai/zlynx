from typing import NamedTuple, Any
import optax
import jax
import jax.numpy as jnp
import numpy as np
import inspect

from .trainer import TrainerConfig


_randomized_svd = None


def _require_randomized_svd():
    global _randomized_svd
    if _randomized_svd is None:
        try:
            from sklearn.utils.extmath import randomized_svd as _imported_randomized_svd
        except ImportError as e:
            raise ImportError(
                "GaLore requires scikit-learn for randomized SVD. "
                "Install it with `pip install -U scikit-learn` to use `galore_*` optimizers."
            ) from e
        _randomized_svd = _imported_randomized_svd
    return _randomized_svd


CORE_OPTIMIZERS = {
    "adabelief", "adadelta", "adafactor", "adagrad", "adam", "adamax",
    "adamaxw", "adamw", "adan", "amsgrad", "fromage", "lamb", "lars",
    "lbfgs", "lion", "lookahead", "nadam", "nadamw", "noisy_sgd",
    "novograd", "optimistic_adam", "optimistic_adam_v2",
    "optimistic_gradient_descent", "polyak_sgd", "radam", "rmsprop",
    "rprop", "sgd", "sign_sgd", "sm3", "yogi",
}

CONTRIB_OPTIMIZERS = {
    "ademamix", "dadapt_adamw", "dog", "dowg", "dpsgd", "madgrad",
    "mechanize", "momo", "momo_adam", "muon", "prodigy", "sam",
}


def get_optimizer(config: TrainerConfig):
    name_or_fn = config.optimizer

    if callable(name_or_fn):
        return name_or_fn

    if not isinstance(name_or_fn, str):
        raise TypeError(
            f"`optimizer` must be a callable or string, got {type(name_or_fn).__name__}"
        )

    name = name_or_fn.removeprefix("galore_").lower()

    if name in CORE_OPTIMIZERS:
        return getattr(optax, name)

    contrib = getattr(optax, "contrib", None)
    if contrib is not None and name in CONTRIB_OPTIMIZERS and hasattr(contrib, name):
        return getattr(contrib, name)

    available = ", ".join(sorted(CORE_OPTIMIZERS | CONTRIB_OPTIMIZERS))
    raise ValueError(
        f"Unknown optimizer: {name_or_fn!r}. "
        f"Available optimizers: {available}"
    )


SCHEDULERS = {
    "constant": lambda lr, total, warmup=0, **kw: optax.constant_schedule(
        value=kw.get("value", lr),
    ),

    "linear": lambda lr, total, warmup=0, **kw: optax.linear_schedule(
        init_value=kw.get("init_value", lr),
        end_value=kw.get("end_value", 0.0),
        transition_steps=kw.get("transition_steps", total),
    ),

    "cosine": lambda lr, total, warmup=0, **kw: optax.cosine_decay_schedule(
        init_value=kw.get("init_value", lr),
        decay_steps=kw.get("decay_steps", total),
        alpha=kw.get("alpha", 0.0),
        exponent=kw.get("exponent", 1.0),
    ),

    "cosine_onecycle": lambda lr, total, warmup=0, **kw: optax.cosine_onecycle_schedule(
        transition_steps=kw.get("transition_steps", total),
        peak_value=kw.get("peak_value", lr),
        pct_start=kw.get("pct_start", 0.3),
        div_factor=kw.get("div_factor", 25.0),
        final_div_factor=kw.get("final_div_factor", 10000.0),
    ),

    "exponential": lambda lr, total, warmup=0, **kw: optax.exponential_decay(
        init_value=kw.get("init_value", lr),
        transition_steps=kw.get("transition_steps", total),
        decay_rate=kw.get("decay_rate", 0.99),
        transition_begin=kw.get("transition_begin", 0),
        staircase=kw.get("staircase", False),
        end_value=kw.get("end_value", None),
    ),

    "join": lambda lr, total, warmup=0, **kw: optax.join_schedules(
        schedules=kw["schedules"],
        boundaries=kw["boundaries"],
    ),

    "linear_onecycle": lambda lr, total, warmup=0, **kw: optax.linear_onecycle_schedule(
        transition_steps=kw.get("transition_steps", total),
        peak_value=kw.get("peak_value", lr),
        pct_start=kw.get("pct_start", 0.3),
        pct_final=kw.get("pct_final", 0.85),
        div_factor=kw.get("div_factor", 25.0),
        final_div_factor=kw.get("final_div_factor", 10000.0),
    ),

    "piecewise_constant": lambda lr, total, warmup=0, **kw: optax.piecewise_constant_schedule(
        init_value=kw.get("init_value", lr),
        boundaries_and_scales=kw.get("boundaries_and_scales", {}),
    ),

    "piecewise_interpolate": lambda lr, total, warmup=0, **kw: optax.piecewise_interpolate_schedule(
        interpolate_type=kw.get("interpolate_type", "linear"),
        init_value=kw.get("init_value", lr),
        boundaries_and_scales=kw.get("boundaries_and_scales", {}),
    ),

    "polynomial": lambda lr, total, warmup=0, **kw: optax.polynomial_schedule(
        init_value=kw.get("init_value", lr),
        end_value=kw.get("end_value", 0.0),
        power=kw.get("power", 1.0),
        transition_steps=kw.get("transition_steps", total),
        transition_begin=kw.get("transition_begin", 0),
    ),

    "sgdr": lambda lr, total, warmup=0, **kw: optax.sgdr_schedule(
        cosine_kwargs=kw["cosine_kwargs"],
    ),

    "warmup_constant": lambda lr, total, warmup=0, **kw: optax.warmup_constant_schedule(
        init_value=kw.get("init_value", 0.0),
        peak_value=kw.get("peak_value", lr),
        warmup_steps=kw.get("warmup_steps", warmup),
    ),

    "warmup_cosine_decay": lambda lr, total, warmup=0, **kw: optax.warmup_cosine_decay_schedule(
        init_value=kw.get("init_value", 0.0),
        peak_value=kw.get("peak_value", lr),
        warmup_steps=kw.get("warmup_steps", warmup),
        decay_steps=kw.get("decay_steps", total),
        end_value=kw.get("end_value", 0.0),
        exponent=kw.get("exponent", 1.0),
    ),

    "warmup_exponential_decay": lambda lr, total, warmup=0, **kw: optax.warmup_exponential_decay_schedule(
        init_value=kw.get("init_value", 0.0),
        peak_value=kw.get("peak_value", lr),
        warmup_steps=kw.get("warmup_steps", warmup),
        transition_steps=kw.get("transition_steps", max(total - warmup, 1)),
        decay_rate=kw.get("decay_rate", 0.99),
        transition_begin=kw.get("transition_begin", 0),
        staircase=kw.get("staircase", False),
        end_value=kw.get("end_value", None),
    ),
}

def get_scheduler(config: TrainerConfig):
    if callable(config.lr_scheduler):
        schedule_fn = config.lr_scheduler
    elif isinstance(config.lr_scheduler, str):
        if config.lr_scheduler not in SCHEDULERS:
            available = ", ".join(sorted(SCHEDULERS))
            raise ValueError(
                f"Unknown lr_scheduler: {config.lr_scheduler!r}. "
                f"Available schedulers: {available}"
            )
        schedule_fn = SCHEDULERS[config.lr_scheduler]
    else:
        raise TypeError(
            f"`lr_scheduler` must be a callable or string, got "
            f"{type(config.lr_scheduler).__name__}"
        )
    return schedule_fn


class GaloreState(NamedTuple):
    inner_state: Any
    projector: Any
    step: jnp.ndarray

def galore_wrapper(inner_opt: optax.GradientTransformation, r: int = 128, update_proj_gap: int = 200, scale: float = 1.0):
    """
    Wraps an Optax optimizer to apply Gradient Low-Rank Projection (GaLore).
    This reduces the memory footprint of optimizer states (like Adam's moments)
    by projecting gradients into a low-rank subspace.

    SVD-based projector updates are triggered automatically every `update_proj_gap`
    steps inside update_fn, which runs outside JIT in the trainer.
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
        # Update projectors via SVD every update_proj_gap steps (runs outside JIT)
        if params is not None and int(state.step) % update_proj_gap == 0:
            state = update_galore_projectors(state, updates, params, r=r)

        # Project gradients down using current projectors
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
    randomized_svd = _require_randomized_svd()

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
        grad_np = np.asarray(grad.astype(jnp.float32))
        U, S, Vh = randomized_svd(grad_np, n_components=r, random_state=None)
        if m < n:
            new_proj_flat.append(jnp.asarray(Vh.T).astype(grad.dtype))
        else:
            new_proj_flat.append(jnp.asarray(U).astype(grad.dtype))

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



def build_optimizer(config: "TrainerConfig", total_steps: int):
    """
    Build an optax optimizer chain from TrainerConfig.

    some optimizer 
    """
    warmup = config.warmup_steps or int(config.warmup_ratio * total_steps)

    schedule_fn = get_scheduler(config)
    schedule = schedule_fn(config.learning_rate, total_steps, warmup)

    opt_fn = get_optimizer(config)
    sig = inspect.signature(opt_fn)

    inner_kwargs = {
        k: v for k, v in config.optimizer_kwargs.items()
        if not k.startswith("galore_")
    }

    kwargs = dict(inner_kwargs)
    if "learning_rate" in sig.parameters:
        kwargs["learning_rate"] = schedule

    inner = opt_fn(**kwargs)

    # Wrap only the base optimizer with GaLore — before weight decay
    is_galore = isinstance(config.optimizer, str) and config.optimizer.startswith("galore_")
    if is_galore:
        _require_randomized_svd()
        galore_kwargs = {
            k.removeprefix("galore_"): v
            for k, v in config.optimizer_kwargs.items()
            if k.startswith("galore_")
        }
        inner = galore_wrapper(
            inner,
            r=galore_kwargs.get("r", 128),
            update_proj_gap=galore_kwargs.get("update_proj_gap", 200),
            scale=galore_kwargs.get("scale", 1.0),
        )

    if config.weight_decay:
        opt = optax.chain(
            optax.add_decayed_weights(config.weight_decay),
            inner,
        )
    else:
        opt = inner

    opt = optax.with_extra_args_support(opt)
    return opt, schedule
