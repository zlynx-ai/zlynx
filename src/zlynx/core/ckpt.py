from flax import nnx
from typing import Optional, Dict, Literal
import os, json, logging, shutil, jax, jax.numpy as jnp
from safetensors.flax import save_file, load_file
from pathlib import Path

def _load_safetensors(state: nnx.State, path: str, module_map: Optional[Dict] = None) -> nnx.State:
        if module_map is None:
            module_map = {}

        index_path = os.path.join(path, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_data = json.load(f)

        weight_map = index_data["weight_map"]

        files_to_load = {}
        for k_str, file_name in weight_map.items():
            if file_name not in files_to_load:
                files_to_load[file_name] = []
            files_to_load[file_name].append(k_str)

        current_state = nnx.to_flat_state(state)
        current_state_dict = dict(current_state) 
        new_state = []
        not_found_some = None

        for file_name, keys_in_file in files_to_load.items():
            shard_path = os.path.join(path, file_name)
            shard_data = load_file(shard_path) 
            
            for k_str in keys_in_file:
                k_tuple = tuple(
                    int(m) if m.isdigit() else module_map.get(m, m) 
                    for m in k_str.split(".")
                )
                
                if k_tuple in current_state_dict:
                    target_var = current_state_dict[k_tuple]
                    instance = type(target_var)
                    
                    sharding = getattr(target_var, "sharding", None) 
                    
                    value = shard_data[k_str] 
                    
                    if sharding is not None:
                        device_value = jax.device_put(instance(value), sharding)
                    else:
                        device_value = jax.device_put(instance(value))
                        
                    new_state.append((k_tuple, device_value))
                else:
                    not_found_some = True
                    print(f"{k_str} not found in this model.")
            
            del shard_data

            if not_found_some is not None and not_found_some:
                print(" some modules are not found in this model you may can try to change the module name with module_map.")
                print(" e.g. module_map = {'target_module': 'name_to_change',}")

        nnx.update(state, nnx.FlatState(new_state, sort=False).to_nested_state())
        return state


def _save_safetensors(model: nnx.Module, path: str | Path, max_shard_size_gb: float=3.0) -> None:
        
        os.makedirs(path, exist_ok=True)

        state = nnx.state(model)

        flat_state = nnx.to_flat_state(state=state)

        max_shard_size = int(max_shard_size_gb * 1024**3)
        
        shards = []
        current_shard = {}
        current_size = 0

        for k, v in dict(flat_state).items():
            k = ".".join(map(str, k))

            if not hasattr(v, "shape") or len(v.shape) == 0:
                v = jnp.array(v)
                
            v_size = v.nbytes

            if current_size + v_size > max_shard_size and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
                
            current_shard[k] = v
            current_size += v_size

        if current_shard:
            shards.append(current_shard)

        num_shards = len(shards)
        weight_map = {}
        total_size = 0

        for i, shard in enumerate(shards, 1):
            file_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
            save_path = os.path.join(path, file_name)
            
            save_file(shard, save_path)
            
            for k, v in shard.items():
                weight_map[k] = file_name
                total_size += v.nbytes

        index_data = {
            "metadata": {
                "total_size": total_size
            },
            "weight_map": weight_map
        }

        with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f, indent=2)


def _save_orbax(model: nnx.Module, path: str | Path) -> None:
    from orbax import checkpoint as ocp
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    state = nnx.state(model)
    ckpter = ocp.StandardCheckpointer()
    staging_path = path / "_orbax_checkpoint"
    if staging_path.exists():
        shutil.rmtree(staging_path)
    ckpter.save(staging_path, state)
    ckpter.wait_until_finished()
    for child in staging_path.iterdir():
        dest = path / child.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(child), str(dest))
    staging_path.rmdir()


def _save_npz(model: nnx.Module, path: str | Path) -> None:
    import numpy as np
    os.makedirs(path, exist_ok=True)
    flat_state = nnx.to_flat_state(nnx.state(model))
    arrays = {".".join(map(str, k)): np.asarray(v) for k, v in dict(flat_state).items()}
    np.savez(os.path.join(path, "model.npz"), **arrays)


def _save_msgpack(model: nnx.Module, path: str | Path) -> None:
    from flax import serialization
    os.makedirs(path, exist_ok=True)
    state_dict = nnx.to_pure_dict(nnx.state(model))
    with open(os.path.join(path, "model.msgpack"), "wb") as f:
        f.write(serialization.msgpack_serialize(state_dict))


def _save_ckpt(
    model: nnx.Module,
    path: str | Path,
    fmt: Literal["orbax", "safetensors", "npz", "msgpack"],
    max_shard_size_gb: float = 3.0,
) -> None:
    if fmt == "orbax":
        _save_orbax(model, path)
    elif fmt == "safetensors":
        _save_safetensors(model, path, max_shard_size_gb=max_shard_size_gb)
    elif fmt == "npz":
        _save_npz(model, path)
    elif fmt == "msgpack":
        _save_msgpack(model, path)
    else:
        raise ValueError(f"unknown checkpoint format: {fmt!r}")


def _load_orbax(state: nnx.State, path: str | Path) -> nnx.State:
    from orbax import checkpoint as ocp
    ckpter = ocp.StandardCheckpointer()
    try:
        state = ckpter.restore(Path(path), state)
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("topology", "mesh", "sharding", "device")):
            logging.warning(
                f"orbax restore failed with sharded template ({e}); "
                "retrying with sharding stripped and re-applying requested sharding after load."
            )
            shardings = jax.tree.map(
                lambda x: getattr(x, "sharding", None),
                state,
                is_leaf=lambda x: hasattr(x, "sharding"),
            )
            stripped = jax.tree.map(
                lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
                if hasattr(x, "shape") and hasattr(x, "dtype") else x,
                state,
            )
            restored = ckpter.restore(Path(path), stripped)
            state = jax.tree.map(
                lambda v, s: jax.device_put(v, s) if s is not None else v,
                restored,
                shardings,
            )
        else:
            raise
    ckpter.wait_until_finished()
    return state


def _load_npz(state: nnx.State, path: str | Path, module_map: Optional[Dict] = None) -> nnx.State:
    import numpy as np
    module_map = module_map or {}

    data = np.load(os.path.join(str(path), "model.npz"))

    current_state_dict = dict(nnx.to_flat_state(state))
    new_state = []
    not_found_some = False

    for k_str in data.files:
        k_tuple = tuple(
            int(m) if m.isdigit() else module_map.get(m, m)
            for m in k_str.split(".")
        )
        if k_tuple in current_state_dict:
            target_var = current_state_dict[k_tuple]
            instance = type(target_var)
            sharding = getattr(target_var, "sharding", None)
            value = jnp.asarray(data[k_str])
            if sharding is not None:
                device_value = jax.device_put(instance(value), sharding)
            else:
                device_value = jax.device_put(instance(value))
            new_state.append((k_tuple, device_value))
        else:
            not_found_some = True
            print(f"{k_str} not found in this model.")

    if not_found_some:
        print(" some modules are not found in this model you may can try to change the module name with module_map.")

    nnx.update(state, nnx.FlatState(new_state, sort=False).to_nested_state())
    return state


def _load_msgpack(state: nnx.State, path: str | Path) -> nnx.State:
    from flax import serialization
    with open(os.path.join(str(path), "model.msgpack"), "rb") as f:
        restored = serialization.msgpack_restore(f.read())
    return serialization.from_state_dict(state, restored)


def _load_ckpt(
    state: nnx.State,
    path: str | Path,
    fmt: Literal["orbax", "safetensors", "npz", "msgpack"],
    module_map: Optional[Dict] = None,
) -> nnx.State:
    if fmt == "orbax":
        return _load_orbax(state, path)
    elif fmt == "safetensors":
        return _load_safetensors(state, path, module_map)
    elif fmt == "npz":
        return _load_npz(state, path, module_map)
    elif fmt == "msgpack":
        return _load_msgpack(state, path)
    else:
        raise ValueError(f"unknown checkpoint format: {fmt!r}")
