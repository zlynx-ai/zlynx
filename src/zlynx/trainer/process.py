

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import datasets
import grain
import types
from typing import Dict

from .trainer_config import TrainerConfig
from .data import grain_from_source, ColumnarDictDataset
from ..utils import param_bytes

def apply_remat(model, *, skip_root=True):
    """
    Apply `nnx.remat` to the `__call__` method of all modules in a model.

    This function walks through the module tree and wraps each module's
    `__call__` with `nnx.remat` so intermediate activations can be
    recomputed during backpropagation instead of stored in memory.

    Args:
        model:
            The root model whose submodules will be modified in place.

        skip_root:
            Whether to skip wrapping the root module itself.
            If `True`, only child modules are wrapped.

    Returns:
        The same model instance, with module `__call__` methods wrapped
        by `nnx.remat`.
    """

    for path, module in nnx.iter_modules(model):
        if skip_root and len(path) == 0:
            continue

        cls_call = type(module).__call__

        def wrapped(self, *args, _cls_call=cls_call, **kwargs):
            return nnx.remat(_cls_call)(self, *args, **kwargs)

        module.__call__ = types.MethodType(wrapped, module)

    return model


def process_model(model, config: TrainerConfig):
    """Apply sharding/placement and gradient checkpointing based on TrainerConfig.

    Args:
        model: An nnx.Module to shard or place.
        config: TrainerConfig with sharding settings.
            sharding="auto": DDP if model fits on one device with >=1.5GB headroom, FSDP otherwise.
            sharding=None: do not change sharding (assumes user applied custom routing/sharding).
            sharding=False: place on first device only (single-device).
            sharding=<int>: place on device with given ID.
            sharding="ddp": replicate across all devices (Distributed Data Parallel).
            sharding="fsdp": shard params across all devices.
            # future: sharding="tp": tensor parallelism.

    Returns:
        The model, placed/sharded according to config.
    """
    devices = jax.devices()
    sharding = config.sharding

    # ── custom sharding (skip) ──
    if sharding is None:
        return model, config

    # ── explicit device ID ──
    if isinstance(sharding, int):
        if sharding >= len(devices):
            raise ValueError(
                f"Device ID {sharding} out of range. "
                f"Available devices: {len(devices)} (IDs 0-{len(devices) - 1})"
            )
        target = jax.devices()[sharding]
        state = nnx.state(model)
        state = jax.device_put(state, target)
        nnx.update(model, state)
        return model, config

    # ── single-device (no sharding) ──
    if sharding is False:
        target = devices[0]
        state = nnx.state(model)
        state = jax.device_put(state, target)
        nnx.update(model, state)
        return model, config

    # ── auto: decide based on memory ──
    if sharding == "auto":
        model_bytes = param_bytes(model)
        # check if model fits on a single device with headroom
        try:
            device_mem = devices[0].memory_stats()
            available = device_mem.get("bytes_limit", float("inf"))
        except Exception:
            available = float("inf")

        headroom = 1.5 * (1024 ** 3)  # 1.5 GiB
        if model_bytes + headroom < available and len(devices) > 1:
            sharding = "ddp"
            config.sharding = sharding
        elif len(devices) > 1:
            sharding = "fsdp"
            config.sharding = sharding
        else:
            # single device, nothing to do
            config.sharding = False
            return model, config

    # ── data parallelism: replicate across all devices ──
    if sharding == "ddp":
        mesh = jax.sharding.Mesh(np.array(devices), ("ddp",))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        state = nnx.state(model)
        state = jax.device_put(state, replicated)
        nnx.update(model, state)
        return model, config

    # ── FSDP: shard params across devices ──
    if sharding == "fsdp":
        mesh = jax.sharding.Mesh(np.array(devices), ("fsdp",))
        sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp"))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        state = nnx.state(model)

        def shard_param(p):
            if not isinstance(p, (jax.Array, np.ndarray)):
                return p
            if p.ndim >= 2:
                return jax.device_put(p, sharded)
            return jax.device_put(p, replicated)

        state = jax.tree.map(shard_param, state)
        nnx.update(model, state)
        return model, config
    
    # ── tensor parallelism ──


    raise ValueError(f"Unknown sharding strategy: {sharding}")


def load_source(dataset, config: TrainerConfig | None = None, is_eval=False):
    """
    Load a dataset source from multiple possible sources.

    If `dataset` is a string, this function attempts to resolve it in the
    following order:

    1. Use `config.load_dataset_fn` if provided.
    2. If the path looks local, try `datasets.load_from_disk`.
    3. Otherwise, or if local loading fails, try `datasets.load_dataset`.

    If `config.dataset_hub` is not `None`, the dataset is loaded through
    `datasets.load_dataset` directly.

    Args:
        dataset:
            Either a dataset object or a string identifying a dataset name
            or local path.

        config:
            Trainer configuration containing dataset loading options such as
            subset name, split names, streaming mode, custom loader function,
            and dataset hub settings.

        is_eval:
            Whether to load the evaluation split instead of the training split.

    Returns:
        The resolved dataset object.

    Notes:
        - Local paths are detected from strings starting with `"."` or `"/"`.
        - When `is_eval=True`, `config.eval_split` is used.
          Otherwise, `config.train_split` is used.
    """

    if isinstance(dataset, str):
        load_dataset_fn = config.load_eval_dataset_fn if is_eval else config.load_train_dataset_fn 
        is_local = dataset.startswith((".", "/"))
        if load_dataset_fn is not None:
            dataset = load_dataset_fn(dataset)
        elif is_local:
            try:
                dataset = datasets.load_from_disk(dataset)
            except Exception as e:
                try:
                    dataset = datasets.load_dataset(
                        dataset,
                        name=config.eval_subset if is_eval else config.train_subset,
                        split=config.eval_split if is_eval else config.train_split,
                        streaming=config.eval_streaming \
                            if is_eval and config.eval_streaming is not None else config.train_streaming \
                                if not is_eval and config.train_streaming is not None else config.streaming,
                    )
                except Exception as e:
                    print(e)
        else:
            dataset = datasets.load_dataset(
                dataset,
                name=config.eval_subset if is_eval else config.train_subset,
                split=config.eval_split if is_eval else config.train_split,
                streaming=config.eval_streaming \
                    if is_eval and config.eval_streaming is not None else config.train_streaming \
                        if not is_eval and config.train_streaming is not None else config.streaming,
            )

    elif isinstance(dataset, Dict):
        dataset = ColumnarDictDataset(dataset)

    return dataset

def process_dataset(dataset, config: TrainerConfig, is_eval=False):
    """
    Prepare a Grain data pipeline from a dataset source.

    This function resolves the dataset source, determines the number of
    examples when available, builds `grain.ReadOptions`, and creates a
    batched pipeline with `grain_from_source`.

    Args:
        dataset:
            Input dataset or dataset identifier accepted by `load_source`.

        config:
            Trainer configuration containing data pipeline options such as
            number of threads, prefetch buffer size, batch size, and seed.

        is_eval:
            Whether the dataset is being prepared for evaluation.

    Returns:
        A tuple `(pipeline, num_examples)` where:
            - `pipeline` is the Grain pipeline created from the source.
            - `num_examples` is the dataset length if available, otherwise `None`.
    """

    source = load_source(dataset, config, is_eval)
    num_examples = len(source) if hasattr(source, '__len__') else None

    read_opts = grain.ReadOptions(
        num_threads=config.num_threads,
        prefetch_buffer_size=config.prefetch_buffer_size,
    )

    seed = config.seed if config.seed is not None else 42

    pipeline = grain_from_source(
        source,
        read_opts=read_opts,
        batch_size=config.batch_size,
        seed=seed
    )

    return pipeline, num_examples
