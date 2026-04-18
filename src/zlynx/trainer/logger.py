from __future__ import annotations

from typing import Any
from pathlib import Path
import json
import jax 
from .trainer_config import TrainerConfig
import datetime


def _detect_sharding(model) -> str | None:
    """Return a short description of actual param sharding, or None if single-device."""
    try:
        from flax import nnx
        leaves = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    except Exception:
        return None
    for leaf in leaves:
        s = getattr(leaf, "sharding", None)
        if s is None:
            continue
        devs = getattr(s, "device_set", None)
        if devs is None or len(devs) <= 1:
            continue
        spec = getattr(s, "spec", None)
        if spec is not None and any(p is not None for p in spec):
            axes = ",".join(str(p) if p is not None else "·" for p in spec)
            return f"partitioned[{axes}] across {len(devs)} devices"
        return f"replicated across {len(devs)} devices"
    return None


def _get_training_banner_args(config: TrainerConfig, total_steps: int, model: Any = None) -> dict[str, Any]:
    devices = jax.devices()
    num_devices = len(devices)
    device = devices[0] if devices else None
    device_name = getattr(device, "device_kind", str(device)) if device is not None else "-"
    sharding = config.sharding

    if sharding:
        if isinstance(sharding, int) and devices:
            device = devices[sharding]
            num_devices = 1
            sharding_display = f"{device_name}[{sharding}]" if device_name != "-" else f"{sharding}"

        elif sharding is None:
            sharding_display = "None"

        else:
            sharding_display = str(sharding)

    else:
        if sharding is False:
            num_devices = 1

        sharding_display = f"{sharding}"

    if model is not None:
        detected = _detect_sharding(model)
        if detected is not None:
            sharding_display = f"{sharding_display} → {detected}"

    batch = config.batch_size
    grad_accum = config.gradient_accumulation_steps

    if sharding == "ddp":
        replicas = num_devices
    else:
        replicas = 1

    effective_batch = batch * replicas * grad_accum
    parts = [f"{batch}"]
    if replicas > 1:
        parts.append(f"{replicas} replicas")
    if grad_accum > 1:
        parts.append(f"{grad_accum} accum")
    effective_str = " × ".join(parts) + f" = {effective_batch}" if len(parts) > 1 else f"{effective_batch}"

    warmup = config.warmup_steps or (
        int(config.warmup_ratio * total_steps) if config.warmup_ratio else 0
    )
    scheduler_str = config.lr_scheduler + (f" (warmup {warmup})" if warmup else "")

    opt_str = str(config.optimizer)
    if config.optimizer_kwargs:
        pad = " " * (16 + 3)  # key_width + " : "
        prefix = (
            f"{config.optimizer.split('_', 1)[0]}_"
            if isinstance(config.optimizer, str) and "_" in config.optimizer
            else ""
        )
        extras = f",\n{pad}".join(
            f"{k.removeprefix(prefix)}={v}" for k, v in config.optimizer_kwargs.items()
        )
        opt_str = f"{opt_str}\n{pad}{extras}"
    if config.weight_decay:
        opt_str += f"  wd={config.weight_decay}"

    return {
        "num_devices": num_devices,
        "device_name": device_name,
        "sharding": sharding_display,
        "remat": config.remat,
        "batch": batch,
        "grad_accum": grad_accum,
        "effective_batch": effective_str,
        "optimizer": opt_str,
        "learning_rate": config.learning_rate,
        "scheduler": scheduler_str,
        "epochs": config.num_epochs,
        "max_steps": total_steps,
        "output_dir": config.output_dir,
    }

class TrainingMetric:
    def __init__(self, model_name: str, config: TrainerConfig):
        start_date = datetime.datetime.now()
        self.config = config
        self.training_metrics = {
            "model": model_name,
            "start_date": start_date,
            "end_date": datetime.datetime.now(),
            "optimizer": config.optimizer,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "lr_scheduler": config.lr_scheduler,
            "warmup_steps": config.warmup_steps,
            "warmup_ratio": config.warmup_ratio,
            "max_steps": config.max_steps,
            "num_epochs": config.num_epochs,
            "remat": config.remat,
            "sharding": config.sharding,
            "steps": {},
        }
    
    def update_training_metrics(self, step, metrics: dict, is_eval = False):
        cp_metrics = metrics.copy()
        if is_eval:
            cp_metrics.pop("eval_step")
            if "eval" not in self.training_metrics:
                self.training_metrics["eval"] = {}
            self.training_metrics["eval"][step] = cp_metrics
        else:
            cp_metrics.pop("step")
            self.training_metrics["steps"][step] = cp_metrics
    
    def save_training_metrics(self):
        model_path = Path(self.config.output_dir).resolve()
        self.training_metrics['end_date'] = datetime.datetime.now()
        with open(model_path / "training_metrics.json", 'w') as f:
            json.dump(self.training_metrics, f, default=str, indent=2)



def format_num(n):
    if n is None:
        return "-"
    n = float(n)
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(int(n))


def format_lr(x):
    if x is None:
        return "-"
    return f"{x:.2e}"


def format_flag(x):
    if x is None:
        return "-"
    return "enabled" if x else "disabled"


def build_training_banner(
    model_name,
    total_params,
    trainable_params,
    num_devices="-",
    device_name="-",
    sharding="-",
    remat=None,
    batch=None,
    grad_accum=None,
    effective_batch=None,
    optimizer="-",
    learning_rate=None,
    scheduler=None,
    epochs=None,
    max_steps=None,
    output_dir=None,
):
    width = 64
    key_width = 16
    title = " Zlynx "

    def format_num(n):
        if n is None:
            return "-"
        n = float(n)
        if n >= 1e12:
            return f"{n / 1e12:.2f}T"
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        if n >= 1e6:
            return f"{n / 1e6:.2f}M"
        if n >= 1e3:
            return f"{n / 1e3:.2f}K"
        return str(int(n))

    def format_lr(x):
        if x is None:
            return "-"
        return f"{x:.2e}"

    def format_flag(x):
        if x is None:
            return "-"
        return "enabled" if x else "disabled"

    def row(key, value):
        return f"{key:<{key_width}} : {value}"

    top = f"{'═' * ((width - len(title)) // 2)}{title}{'═' * (width - ((width - len(title)) // 2) - len(title))}"
    bottom = "═" * width

    lines = [
        top,
        row("Model", model_name),
        row("Parameters", f"{format_num(total_params)} total | {format_num(trainable_params)} trainable"),
        row("Devices", f"{num_devices} x {device_name}"),
        row("Sharding", sharding),
        row("Remat", format_flag(remat)),
        "",
        row("Batch size", f"{batch}" if batch is not None else "-"),
        row("Grad accum", grad_accum if grad_accum is not None else "-"),
        row("Effective batch", effective_batch if effective_batch is not None else "-"),
        "",
        row("Optimizer", optimizer),
        row("Learning rate", format_lr(learning_rate)),
        row("Scheduler", scheduler),
        row("Epochs", epochs if epochs is not None else "-"),
        row("Max steps", max_steps if max_steps is not None and max_steps > 0 else "-"),
        "",
        row("Output dir", output_dir if output_dir else "-"),
        bottom,
    ]
    return "\n".join(lines)

def print_training_banner(**kw):
    print(build_training_banner(**kw))


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

        self.is_in_notebook = self.in_notebook()
        if self.is_in_notebook:
            self.rows = []
            self.eval_rows = []
            self.display_id = None
            self.eval_display_id = None
        

    def in_notebook(self):
        try:
            from IPython import get_ipython
            import pandas as pd
            shell = get_ipython()
            if shell is None:
                return False
            return shell.__class__.__name__ in ("ZMQInteractiveShell",)
        except Exception:
            return False
        
    def log(self, metrics: dict, step: int, is_eval: bool=False):
        """Log metrics to all active backends."""
        if self.is_in_notebook:
            from IPython.display import display, update_display, HTML
            import pandas as pd
            if is_eval:
                self.eval_rows.append(metrics)
                df = pd.DataFrame(self.eval_rows).set_index("eval_step")
                if self.eval_display_id is None:
                    handle = display(df, display_id=True)
                    self.eval_display_id = handle.display_id
                else:
                    update_display(df, display_id=self.eval_display_id)
            else:
                self.rows.append(metrics)
                df = pd.DataFrame(self.rows).set_index("step")
                if self.display_id is None:
                    handle = display(df, display_id=True)
                    self.display_id = handle.display_id
                else:
                    update_display(df, display_id=self.display_id)
        else:
            from tqdm import tqdm
            tqdm.write(
                str({k: f"{v:.4e}" if isinstance(v, float) and v < 10e-3 else \
                     round(v, 4) if isinstance(v, float) else \
                        v for k, v in metrics.items()})
            )

        if "tensorboard" in self.backends and self._tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step)
            self._tb_writer.flush()

        if "wandb" in self.backends:
            import wandb
            wandb.log(metrics, step=step)
        
    def close(self):
        if self._tb_writer:
            self._tb_writer.close()
        if "wandb" in self.backends:
            import wandb
            if wandb.run:
                wandb.finish()

