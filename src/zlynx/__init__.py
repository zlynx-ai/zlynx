
from .core import Z, LanguageConfig, LanguageModel, CausalLMOutput, Config, ModelOutput
from .trainer import Trainer, TrainerConfig

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("zlynx")
except PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = ["Z", "LanguageConfig", "LanguageModel", "CausalLMOutput", "Config", "ModelOutput", "Trainer", "TrainerConfig"]