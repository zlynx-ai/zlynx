from __future__ import annotations

from typing import Any
from pathlib import Path
import json
import jax 
from .trainer_config import TrainerConfig
import datetime


def _get_training_banner_args(config: TrainerConfig, model, total_steps: int) -> dict[str, Any]:
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
        num_devices = 1
        sharding_display = f"{sharding}"

    per_device_batch = config.per_device_batch_size
    grad_accum = config.gradient_accumulation_steps
    global_batch = per_device_batch * max(num_devices, 1) * grad_accum
    precision = (
        config.dtype
        if config.grad_dtype in (None, config.dtype)
        else f"{config.dtype} params / {config.grad_dtype} grads"
    )

    return {
        "num_devices": num_devices,
        "device_name": device_name,
        "sharding": sharding_display,
        "precision": precision,
        "remat": config.remat,
        "train_batch": per_device_batch,
        "grad_accum": grad_accum,
        "global_batch": f"{per_device_batch} x {grad_accum} = {global_batch} / device",
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "scheduler": config.lr_scheduler,
        "epochs": config.num_epochs,
        "max_steps": total_steps,
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
            "per_device_batch_size": config.per_device_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "lr_scheduler": config.lr_scheduler,
            "warmup_steps": config.warmup_steps,
            "warmup_ratio": config.warmup_ratio,
            "max_steps": config.max_steps,
            "num_epochs": config.num_epochs,
            "dtype": config.dtype,
            "grad_dtype": config.dtype if config.grad_dtype is None else config.grad_dtype,
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
    precision="-",
    remat=None,
    train_batch=None,
    grad_accum=None,
    global_batch=None,
    optimizer="-",
    learning_rate=None,
    scheduler=None,
    epochs=None,
    max_steps=None,
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
        row("Precision", precision),
        row("Remat", format_flag(remat)),
        "",
        row("Train batch", f"{train_batch} / device" if train_batch is not None else "-"),
        row("Grad accum", grad_accum if grad_accum is not None else "-"),
        row("Global batch", global_batch if global_batch is not None else "-"),
        "",
        row("Optimizer", optimizer),
        row("Learning rate", format_lr(learning_rate)),
        row("Scheduler", scheduler),
        row("Epochs", epochs if epochs is not None else "-"),
        row("Max steps", max_steps if max_steps is not None else "-"),
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

        # if "json" in backends:
        #     self._json_path = self.output_dir / "train_log.jsonl"
        #     self._json_path.parent.mkdir(parents=True, exist_ok=True)

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

        # if "json" in self.backends and self._json_path:
        #     record = {"step": step, **{k: float(v) if isinstance(v, (int, float, jnp.ndarray)) else v for k, v in metrics.items()}}
        #     with open(self._json_path, "a") as f:
        #         f.write(json.dumps(record) + "\n")

    
        
    def close(self):
        if self._tb_writer:
            self._tb_writer.close()
        if "wandb" in self.backends:
            import wandb
            if wandb.run:
                wandb.finish()

