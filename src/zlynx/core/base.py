

from flax import nnx, serialization, struct
from pathlib import Path
from orbax import checkpoint as ocp
from typing import Literal, List, Dict, Tuple, Set, Optional, Any
from dataclasses import field, is_dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as installed_version
import json
import jax, jax.numpy as jnp
import logging
import shutil
import tempfile
import os
import re
import inspect
from datetime import datetime

from .ckpt import _load_safetensors, _save_safetensors, _save_ckpt, _load_ckpt


def _normalize_dtypes_in_config_dict(config_dict: dict) -> dict:
    """Convert `dtype` / `param_dtype` from dtype objects to str names for JSON serialization."""
    for key in ("dtype", "param_dtype"):
        if key in config_dict and config_dict[key] is not None and not isinstance(config_dict[key], str):
            try:
                config_dict[key] = jnp.dtype(config_dict[key]).name
            except Exception:
                pass
    return config_dict


def _version_key(version: str) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (0, int(token)) if token.isdigit() else (1, token.lower())
        for token in re.findall(r"\d+|[A-Za-z]+", version)
    )


def _require_optional_dependency(
    module_name: str,
    package_name: str,
    min_version: str,
):
    try:
        current_version = installed_version(package_name)
    except PackageNotFoundError as e:
        raise ImportError(
            f"`{package_name}` is required for this operation. "
            f"Install it with `pip install -U {package_name}`."
        ) from e

    if _version_key(current_version) < _version_key(min_version):
        raise ImportError(
            f"`{package_name}` {current_version} is too old for this operation. "
            f"Please update it with `pip install -U {package_name}`."
        )

    try:
        return import_module(module_name)
    except Exception as e:
        raise ImportError(
            f"Failed to import `{package_name}` {current_version}. "
            f"Try updating it with `pip install -U {package_name}`."
        ) from e


def _bundle_directory(src_dir: Path, archive_name: str, archive_format: str) -> Path:
    with tempfile.TemporaryDirectory(dir=src_dir.parent, prefix="zlynx_bundle_") as bundle_dir:
        archive_base = Path(bundle_dir) / archive_name
        archive_path = Path(
            shutil.make_archive(
                str(archive_base),
                archive_format,
                root_dir=src_dir,
            )
        )
        upload_dir = src_dir / "_upload_bundle"
        upload_dir.mkdir(exist_ok=True)
        final_archive_path = upload_dir / archive_path.name
        shutil.move(str(archive_path), str(final_archive_path))
        return final_archive_path


def _maybe_unpack_single_archive(path: Path) -> Path:
    archive_files = [
        p for p in path.iterdir()
        if p.is_file() and (p.suffix == ".zip" or "".join(p.suffixes[-2:]) == ".tar.gz")
    ]
    if len(archive_files) != 1:
        return path

    archive_path = archive_files[0]
    extract_dir = path / "_extracted"
    extract_dir.mkdir(exist_ok=True)
    shutil.unpack_archive(str(archive_path), str(extract_dir))
    return extract_dir


def _caller_globals() -> dict[str, Any]:
    frame = inspect.currentframe()
    try:
        frame = frame.f_back
        while frame is not None:
            if frame.f_code.co_filename != __file__:
                return frame.f_globals
            frame = frame.f_back
        return {}
    finally:
        del frame


def _trace_safe_constructor_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    trace_kwargs = dict(kwargs)

    for name, value in trace_kwargs.items():
        if isinstance(value, nnx.Rngs):
            trace_kwargs[name] = {
                tag: {
                    "key": stream.key.get_value(),
                    "count": stream.count.get_value(),
                }
                for tag, stream in value.items()
            }

    return trace_kwargs


def _materialize_trace_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    materialized = dict(kwargs)

    for name, value in materialized.items():
        if (
            isinstance(value, dict)
            and value
            and all(
                isinstance(stream, dict) and "key" in stream and "count" in stream
                for stream in value.values()
            )
        ):
            rngs = nnx.Rngs({
                tag: stream["key"]
                for tag, stream in value.items()
            })
            for tag, stream in value.items():
                getattr(rngs, tag).count[...] = stream["count"]
            materialized[name] = rngs

    return materialized


class Z(nnx.Module):
    """
    # Z Class

    Base class for model checkpointing (save/load weights).

    ## Quick Start

    ```python
    from zlynx import Z

    class MyModel(Z):
        def __init__(self, config):
            self.config = config
            self.embed = nnx.Embed(num_embeddings=1000, features=256)
            self.linear = nnx.Linear(256, 1000)

    # Save
    model = MyModel(config)
    model.save("./my_model")

    # Load
    model = MyModel.load("./my_model")
    ```

    ## Methods

    | Method | Description |
    |--------|-------------|
    | `save(path)` | Save model weights to disk |
    | `load(path)` | Load model weights (classmethod) |
    | `load_hf(repo_id)` | Load from HuggingFace |
    | `push_hf(repo_id)` | Push to HuggingFace |
    | `load_kaggle(repo_id)` | Load from Kaggle |
    | `push_kaggle(repo_id)` | Push to Kaggle |

    ## Formats

    - `orbax` (default) - JAX native, fast
    - `safetensors` - Cross-platform, compatible with HF

    ## Notes

    - Inherit from `Z` to get save/load for free
    - Requires `config` attribute for saving config
    - Auto-detects architecture when using `Z.load()` directly
    """
    
    def set_config(self, config: Dict[str, Any] | Any) -> None:
        if not isinstance(config, dict) and not is_dataclass(config):
            raise TypeError(
                "`config` must be a dict or a dataclass instance."
            )

        self._config = config

    # return config
    @classmethod
    def load_config(
        cls, path: str | Path, 
        return_dict: bool = False, 
        config_map: Optional[Dict] = None
    ) -> Any:
        
        path = Path(path).resolve()
        caller_globals = _caller_globals()

        if not (path / "config.json").exists(): 
            return None

        with open(path / "config.json", "r") as config_file:
            config_dict: dict = json.load(config_file)

        if config_map is not None:
            config_dict = {config_map[k] if k in config_map else k:v for k, v in config_dict.items()}

        if return_dict:
            return config_dict

        config_class = None
        config_class_name = config_dict.get("config_class", None) 
        
        if config_class_name is not None:
            from .. import model

            # try in-lib model config
            config_class = getattr(model, config_class_name, None)
            
            # try user defined
            if config_class is None:
                config_class = caller_globals.get(config_class_name)

        # still None return as C base config
        if config_class is None:
            from ..core.config import Config
            config_class = type('Config', (Config,), {
                '__annotations__': {**Config.__annotations__, **{k: type(v) for k, v in config_dict.items()}},
                **{k: v if not isinstance(v, (List, Dict, Tuple, Set)) else field(default_factory=lambda v=v: v) for k, v in config_dict.items()}
            })
            config_class = struct.dataclass(config_class)

        return config_class(**config_dict)

    @classmethod
    def load(
        cls, path: str | Path, 
        *args,
        dtype: str | None=None, 
        ignore_local_config: bool = False,
        config_map: Optional[Dict]=None,
        module_map: Optional[Dict]=None,
        sharding: int | str | None=None,
        fmt: Literal["orbax", "safetensors", "npz", "msgpack"]="orbax",
        **kwargs
    ) -> "Z":
        
        """
        Load a model from a checkpoint.

        This method rebuilds the model by calling the class constructor first,
        then restores checkpoint parameters into that instance. Because of that,
        any extra `*args` and `**kwargs` passed to `load()` are forwarded to
        the model constructor (`__init__`), and required constructor arguments
        must still be provided at load time.

        Args:
            path:
                Path to the checkpoint directory or file.

            *args:
                Extra positional arguments forwarded to the model constructor.

            dtype:
                Optionally cast model parameters to a different dtype while loading.

            config:
                Optional model config object to use during reconstruction.
                Some models, such as those in `zlynx.models`, may save and load
                a dataclass-based config together with the checkpoint.

            config_map:
                Optional mapping for config field names when loading checkpoints
                across implementations that use equivalent configs with different
                key names. For example, some Hugging Face-style models may share
                the same logical config structure but use different field names.

            module_map:
                Optional mapping for module / parameter names when loading between
                models with the same architecture but different internal naming.
                For example, one model may use `q_proj` while another uses `w_q`.

            sharding:
                Optionally load the model with sharding applied.

            format:
                Checkpoint format to load. This must be specified when the target
                directory may contain multiple checkpoint formats, such as both
                `orbax` and `safetensors`.

            **kwargs:
                Extra keyword arguments forwarded to the model constructor.

        Returns:
            The loaded model instance.
        """

        path = Path(path).resolve()
        caller_globals = _caller_globals()

        config = None
        if not ignore_local_config:
            config = cls.load_config(path, config_map=config_map)

        if cls is Z:
            
            arch_name = None
            if config is not None:
                arch_name = getattr(config, "architecture", None)
                
            if arch_name is None:
                raise ValueError("Could not determine model architecture.")
            
            from .. import model
            arch = getattr(model, arch_name, None)

            if arch is None:
                arch = caller_globals.get(arch_name)

            if arch is None:
                raise ValueError("Could not determine model architecture.")

        else:
            # allow config = None
            arch = cls
        
        logging.info(f"{arch.__name__} model class obtained")

        if config is None:
            model = nnx.eval_shape(
                lambda: arch(*args, **kwargs)
            )
        else:
            model = nnx.eval_shape(
                lambda: arch(config, *args, **kwargs)
            )
        
        gdef, state = nnx.split(model)

        if sharding is not None:

            # auto sharding
            if sharding == "ddp":
                mesh = jax.sharding.Mesh(jax.devices(), ("data",))
            elif sharding == "fsdp":
                mesh = jax.sharding.Mesh(jax.devices(), ("model",))

            def wrap_with_sharding(leaf):
                shape = leaf.shape
                rank = len(shape)
                shard_dim = shape[0] if rank > 0 else None

                if rank == 0 or sharding == "ddp":
                    spec = jax.sharding.PartitionSpec()
                elif shard_dim is not None and shard_dim % jax.device_count() != 0:
                    spec = jax.sharding.PartitionSpec()
                else:
                    spec = jax.sharding.PartitionSpec("model")

                actual_sharding = jax.sharding.NamedSharding(mesh, spec)

                return jax.ShapeDtypeStruct(
                    shape=leaf.shape, dtype=leaf.dtype, sharding=actual_sharding
                )

            state = jax.tree.map(wrap_with_sharding, state)

        state = _load_ckpt(state, path, fmt=fmt, module_map=module_map)

        if dtype is not None:
            target_dtype = dtype
            def _cast(x):
                if hasattr(x, "dtype") and x.dtype != target_dtype and jnp.issubdtype(x.dtype, jnp.floating):
                    return x.astype(target_dtype)
                return x
            state = jax.tree.map(_cast, state)

        model = nnx.merge(gdef, state)

        return model
    
    def save(
        self, path: str | Path, *,
        fmt: Literal["orbax", "safetensors", "npz", "msgpack"] = "orbax",
        max_shard_size_gb: float = 3.0,
    ) -> None:
        if isinstance(path, str):
            path = Path(path)
        if not path.is_absolute():
            path = path.resolve()

        with tempfile.TemporaryDirectory(dir=path.parent, prefix="zlynx_ckpt_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            try:
                config = getattr(self, "_config", None)

                _save_ckpt(self, tmp_path, fmt=fmt, max_shard_size_gb=max_shard_size_gb)

                if config is not None:
                    with open(tmp_path / "config.json", "w") as config_file:
                        json.dump(_normalize_dtypes_in_config_dict(serialization.to_state_dict(config)), config_file, indent=2)

                if jax.process_index() == 0:
                    shutil.copytree(tmp_path, path, dirs_exist_ok=True)
                    logging.info(f"Save model path {path}.")

            except Exception as e:
                logging.error(e)

    def push_hf(
        self, repo_id: str,
        private: bool = False,
        *,
        fmt: Literal["orbax", "safetensors", "npz", "msgpack"]="safetensors",
        max_shard_size_gb: float = 3.0,
        exist_ok: bool = True,
        **kwargs
    ):
        huggingface_hub = _require_optional_dependency(
            "huggingface_hub",
            "huggingface-hub",
            "1.6.0",
        )
        create_repo = huggingface_hub.create_repo
        upload_folder = huggingface_hub.upload_folder

        repo_type = "model"

        create_repo_keys = {
            "token", "visibility", "resource_group_id", "space_sdk",
            "space_hardware", "space_storage", "space_sleep_time",
            "space_secrets", "space_variables", "space_volumes",
        }
        upload_folder_keys = {
            "path_in_repo", "commit_message", "commit_description", "token",
            "revision", "create_pr", "parent_commit", "allow_patterns",
            "ignore_patterns", "delete_patterns", "run_as_future",
        }

        create_repo_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in create_repo_keys}
        upload_folder_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in upload_folder_keys}
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s) for push_hf: {unknown}")

        repo = create_repo(
            repo_id=repo_id,
            private=private,
            repo_type=repo_type,
            exist_ok=exist_ok,
            **create_repo_kwargs,
        )
        repo_id = repo.repo_id


        with tempfile.TemporaryDirectory(dir=".", prefix="zlynx_ckpt_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            try:
                config = getattr(self, "config", self.kwargs.get("config", None) if hasattr(self, "kwargs") else None)

                _save_ckpt(self, tmp_path, fmt=fmt, max_shard_size_gb=max_shard_size_gb)

                if config is not None:
                    with open(tmp_path / "config.json", "w") as config_file:
                        json.dump(_normalize_dtypes_in_config_dict(serialization.to_state_dict(config)), config_file, indent=2)

                if jax.process_index() == 0:
                    commit_info = upload_folder(
                        folder_path=tmp_path,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        **upload_folder_kwargs,
                    )

                    logging.info(f"Pushed model to HuggingFace https://huggingface.co/{repo_id}")
                    return commit_info

            except Exception as e:
                logging.error(e)

    def push_kaggle(
        self, repo_id: str,
        variation: str="default",
        *,
        fmt: Literal["orbax", "safetensors", "npz", "msgpack"]="safetensors",
        max_shard_size_gb: float=3.0,
        bundle: bool = False,
        archive_format: Literal["zip", "gztar"] = "gztar",
        **kwargs
    ) -> None:
        """
        ### login kaggle:
        ```python
        import kagglehub
        kagglehub.login()
        ```
        """
        kagglehub = _require_optional_dependency(
            "kagglehub",
            "kagglehub",
            "1.0.0",
        )

        date = datetime.now().strftime("%Y-%m-%d")

        with tempfile.TemporaryDirectory(dir=".", prefix="zlynx_ckpt_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            try:
                config = getattr(self, "config", self.kwargs.get("config", None) if hasattr(self, "kwargs") else None)

                _save_ckpt(self, tmp_path, fmt=fmt, max_shard_size_gb=max_shard_size_gb)

                if config is not None:
                    with open(tmp_path / "config.json", "w") as config_file:
                        json.dump(_normalize_dtypes_in_config_dict(serialization.to_state_dict(config)), config_file, indent=2)

                if jax.process_index() == 0:
                    local_model_dir = tmp_path
                    if bundle:
                        archive_name = variation.replace("/", "_") or "model"
                        archive_path = _bundle_directory(tmp_path, archive_name, archive_format)
                        local_model_dir = archive_path.parent
                    kagglehub.model_upload(
                        handle = f"{repo_id}/flax/{variation}",
                        local_model_dir = str(local_model_dir),
                        version_notes = f'Update {date}',
                        **kwargs
                    )

                    logging.info(f"Pushed model to Kaggle https://www.kaggle.com/models/{repo_id}")

            except Exception as e:
                logging.error(e)

    @classmethod
    def load_hf(
        cls, repo_id: str,
        *args,
        dtype: Optional[str]=None,
        config_map: Optional[Dict]=None,
        module_map: Optional[Dict]=None,
        sharding: Optional[Literal["ddp", "fsdp"]]=None,
        fmt: Literal["orbax", "safetensors", "npz", "msgpack"]="safetensors",
        **kwargs
    ) -> "Z":
        huggingface_hub = _require_optional_dependency(
            "huggingface_hub",
            "huggingface-hub",
            "1.6.0",
        )
        snapshot_download = huggingface_hub.snapshot_download

        snapshot_keys = {
            "revision", "cache_dir", "local_dir", "library_name", "library_version",
            "user_agent", "etag_timeout", "force_download", "token", "local_files_only",
            "allow_patterns", "ignore_patterns", "max_workers", "tqdm_class", "headers",
            "endpoint", "dry_run",
        }
        snapshot_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in snapshot_keys}

        local_dir = snapshot_download(
            repo_id=repo_id, repo_type="model",
            **snapshot_kwargs
        )
        path = Path(local_dir).resolve()

        return cls.load(
            path, *args,
            dtype=dtype,
            sharding=sharding,
            fmt=fmt,
            config_map=config_map,
            module_map=module_map,
            **kwargs
        )


    @classmethod
    def load_kaggle(
        cls, repo_id: str,
        variation: str="default",
        *args,
        dtype: Optional[str]=None,
        config_map: Optional[Dict]=None,
        module_map: Optional[Dict]=None,
        sharding: Optional[Literal["ddp", "fsdp"]]=None,
        fmt: Literal["orbax", "safetensors", "npz", "msgpack"]="safetensors",
        **kwargs
    ) -> "Z":
        kagglehub = _require_optional_dependency(
            "kagglehub",
            "kagglehub",
            "1.0.0",
        )

        download_keys = {"path", "force_download", "output_dir"}
        download_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in download_keys}

        local_dir = kagglehub.model_download(
            f"{repo_id}/flax/{variation}",
            **download_kwargs
        )
        path = _maybe_unpack_single_archive(Path(local_dir).resolve())

        return cls.load(
            path, *args,
            dtype=dtype,
            sharding=sharding,
            fmt=fmt,
            config_map=config_map,
            module_map=module_map,
            **kwargs
        )

