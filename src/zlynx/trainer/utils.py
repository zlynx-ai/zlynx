

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from pathlib import Path
import json
import datasets
import grain
from huggingface_hub import hf_hub_download, HfApi
from typing import Optional, Any

from .trainer_config import DatasetConfig, TrainerConfig



def count_params(model) -> int:
    """Count total number of parameters in a model."""
    _, state = nnx.split(model)
    leaves = jax.tree.leaves(state)
    return sum(p.size for p in leaves if hasattr(p, 'size'))


def param_bytes(model, dtype=jnp.float32) -> int:
    """Estimate model memory in bytes for a given dtype."""
    return count_params(model) * jnp.dtype(dtype).itemsize


def apply_gradient_checkpointing(model):
    """Wrap transformer block __call__ methods with jax.checkpoint (remat)
    to recompute activations during backward instead of storing them."""
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        blocks = model.model.blocks
    elif hasattr(model, "blocks"):
        blocks = model.blocks
    else:
        return model

    for block in blocks:
        original_call = block.__call__
        block.__call__ = nnx.remat(original_call)

    return model


def process_model(model, trconfig):
    """Apply sharding/placement and gradient checkpointing based on TrainerConfig.

    Args:
        model: An nnx.Module to shard or place.
        trconfig: TrainerConfig with sharding settings.
            sharding="auto": DDP if model fits on one device with >=1.5GB headroom, FSDP otherwise.
            sharding=None: do not change sharding (assumes user applied custom routing/sharding).
            sharding=False: place on first device only (single-device).
            sharding=<int>: place on device with given ID.
            sharding="dp": replicate across all devices (data parallelism).
            sharding="fsdp": shard params across all devices.
            sharding="tp": tensor parallelism.

    Returns:
        The model, placed/sharded according to config.
    """
    devices = jax.devices()
    sharding = trconfig.sharding

    # ── custom sharding (skip) ──
    if sharding is None:
        return model

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
        return model

    # ── single-device (no sharding) ──
    if sharding is False:
        target = devices[0]
        state = nnx.state(model)
        state = jax.device_put(state, target)
        nnx.update(model, state)
        return model

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
            sharding = "dp"
        elif len(devices) > 1:
            sharding = "fsdp"
        else:
            # single device, nothing to do
            return model

    # ── data parallelism: replicate across all devices ──
    if sharding == "dp":
        mesh = jax.sharding.Mesh(np.array(devices), ("dp",))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        state = nnx.state(model)
        state = jax.device_put(state, replicated)
        nnx.update(model, state)
        return model

    # ── FSDP: shard params across devices ──
    if sharding == "fsdp":
        mesh = jax.sharding.Mesh(np.array(devices), ("fsdp",))
        sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp"))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        state = nnx.state(model)

        def shard_param(p):
            if not hasattr(p, 'ndim'):
                return p
            if p.ndim >= 2:
                return jax.device_put(p, sharded)
            else:
                return jax.device_put(p, replicated)

        state = jax.tree.map(shard_param, state)
        nnx.update(model, state)
        return model

    # ── tensor parallelism ──
    if sharding == "tp":
        mesh = jax.sharding.Mesh(np.array(devices), ("tp",))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        state = nnx.state(model)

        def shard_tp_param(path, p):
            if not hasattr(p, "ndim"):
                return p

            # path is a tuple of nnx tree keys (DictKey, SequenceKey, etc.)
            path_str = ".".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)

            pspec = None
            if "embed_tokens" in path_str and "embedding" in path_str:
                pspec = jax.sharding.PartitionSpec("tp", None)

            elif "lm_head" in path_str and "kernel" in path_str:
                pspec = jax.sharding.PartitionSpec(None, "tp")

            # MLP
            elif "gate_proj" in path_str and ("kernel" in path_str or "bias" in path_str):
                pspec = jax.sharding.PartitionSpec(None, "tp") if "kernel" in path_str else jax.sharding.PartitionSpec("tp")
            elif "up_proj" in path_str and ("kernel" in path_str or "bias" in path_str):
                pspec = jax.sharding.PartitionSpec(None, "tp") if "kernel" in path_str else jax.sharding.PartitionSpec("tp")
            elif "down_proj" in path_str and "kernel" in path_str:
                pspec = jax.sharding.PartitionSpec("tp", None)

            # Attention
            elif "q_proj" in path_str and ("kernel" in path_str or "bias" in path_str):
                pspec = jax.sharding.PartitionSpec(None, "tp") if "kernel" in path_str else jax.sharding.PartitionSpec("tp")
            elif "k_proj" in path_str and ("kernel" in path_str or "bias" in path_str):
                pspec = jax.sharding.PartitionSpec(None, "tp") if "kernel" in path_str else jax.sharding.PartitionSpec("tp")
            elif "v_proj" in path_str and ("kernel" in path_str or "bias" in path_str):
                pspec = jax.sharding.PartitionSpec(None, "tp") if "kernel" in path_str else jax.sharding.PartitionSpec("tp")
            elif "o_proj" in path_str and "kernel" in path_str:
                pspec = jax.sharding.PartitionSpec("tp", None)

            if pspec is not None:
                return jax.device_put(p, jax.sharding.NamedSharding(mesh, pspec))
            else:
                return jax.device_put(p, replicated)

        state = jax.tree_util.tree_map_with_path(shard_tp_param, state)
        nnx.update(model, state)
        return model

    raise ValueError(f"Unknown sharding strategy: {sharding}")


def _apply_checkpointing(model, trconfig):
    if getattr(trconfig, "gradient_checkpointing", False):
        apply_gradient_checkpointing(model)
    return model

def load_dataset(dataset, config: DatasetConfig | None = None):
    """Load a dataset and return a random-access source for grain.

    Accepts:
        str: HF dataset name or local path → loaded via datasets.load_dataset
        datasets.Dataset: used directly (supports __getitem__ + __len__)
        list: used directly as a random-access source
        datasets.IterableDataset: NOT supported (grain MapDataset needs random access)

    Returns:
        A source compatible with grain.MapDataset.source()
    """
    config = config or DatasetConfig()

    if isinstance(dataset, str):
        dataset = datasets.load_dataset(
            dataset,
            name=config.subset,
            split=config.split,
            streaming=config.streaming,
        )

    if isinstance(dataset, datasets.Dataset):
        return dataset
    elif isinstance(dataset, datasets.IterableDataset):
        raise TypeError(
            "Streaming/IterableDataset is not supported for now "
            "(requires random access). Use streaming=False."
        )
    elif isinstance(dataset, list):
        return dataset
    elif isinstance(dataset, dict):
        return datasets.Dataset.from_dict(dataset)
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")



def process_dataset(ds, dsconfig: DatasetConfig, trconfig: TrainerConfig):
    """Build a grain pipeline from a raw dataset source.

    Returns:
        Tuple of (pipeline, num_examples) where pipeline is a grain IterDataset
        and num_examples is the number of examples in the source (for step estimation).
    """
    map_fn = lambda x: dsconfig.preprocessing_fn(x) if dsconfig.preprocessing_fn is not None else x
    fil_fn = lambda x: dsconfig.filter_fn(x) if dsconfig.filter_fn is not None else True
    read_opts = grain.ReadOptions(
        num_threads=dsconfig.num_threads,
        prefetch_buffer_size=dsconfig.prefetch_buffer_size,
    )

    source = load_dataset(ds, dsconfig)
    num_examples = len(source) if hasattr(source, '__len__') else None

    pipeline = grain.MapDataset.source(source)

    if dsconfig.shuffle:
        pipeline = pipeline.shuffle(seed=dsconfig.shuffle_seed)

    pipeline = (
        pipeline
        .map(map_fn)
        .to_iter_dataset(read_opts)
        .filter(fil_fn)
        .batch(batch_size=trconfig.batch_size)
    )

    return pipeline, num_examples

def hf_list_repo_files(
    repo_id: str,
    repo_type: str,
    token: Optional[str]
):
    api = HfApi()

    files = api.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token
    )

    return files


def hf_load_single():
    ...
    

class HFDatasetIterator(grain.DatasetIterator):

  def __init__(self, ds: datasets.IterableDataset):
    self._reader = "<define your reader>"
    self._offset = "<get reader's offset>"

  def __next__(self):
    record = next(self._reader)
    self._record_offset = "<get reader's offset>"
    return record

  def get_state(self) -> dict[str, Any]:
    return {"offset": self._offset}

  def set_state(self, state):
    self._offset = state["offset"]
    self._reader.Seek(self._offset)  # Seeks to the correct offset


class YourCustomIterDataset(grain.IterDataset):

  def __init__(self, ds: datasets.IterableDataset):
    super().__init__()
    self._ds = ds

  def __iter__(self):
    return HFDatasetIterator(self._ds)




# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────

class Logger:
    """Multi-backend logger supporting stdout, TensorBoard, W&B, and JSON."""

    def __init__(self, backends: list[str], output_dir: str, run_name: str | None = None):
        self.backends = backends
        self.output_dir = Path(output_dir)
        self._tb_writer = None
        self._json_path = None

        if "tensorboard" in backends:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.output_dir / "tb_logs"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))

        if "wandb" in backends:
            import wandb
            if not wandb.run:
                wandb.init(project=run_name or "zlynx", name=run_name)

        if "json" in backends:
            self._json_path = self.output_dir / "train_log.jsonl"
            self._json_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict, step: int):
        """Log metrics to all active backends."""
        if "stdout" in self.backends:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

        if "tensorboard" in self.backends and self._tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step)
            self._tb_writer.flush()

        if "wandb" in self.backends:
            import wandb
            wandb.log(metrics, step=step)

        if "json" in self.backends and self._json_path:
            record = {"step": step, **{k: float(v) if isinstance(v, (int, float, jnp.ndarray)) else v for k, v in metrics.items()}}
            with open(self._json_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def close(self):
        if self._tb_writer:
            self._tb_writer.close()
        if "wandb" in self.backends:
            import wandb
            if wandb.run:
                wandb.finish()

