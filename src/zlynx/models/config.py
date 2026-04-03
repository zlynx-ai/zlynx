


from flax import struct



@struct.dataclass
class Cnf:
    arch: str | None = None
    conf: str | None = None
    
    


# @struct.dataclass
# class VisionConfig(C):
#     ...


# @struct.dataclass
# class AudioConfig(C):
#     ...


@struct.dataclass
class LanguageConfig(Cnf):
    vocab_size: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    act_fn: str | None = None
    num_hidden_layers: int | None = None
    norm_eps: float | None = None
    bias: bool | None = None
    dtype: str | None = None
    param_dtype: str | None = None
    use_cache: bool | None = None

    # attn
    attention_head: int | None = None
    kv_head: int | None = None
    head_dim: int | None = None
    attention_bias: bool | None = None

    # rope
    base: float | None = None
    original_max_position_embedding: int | None = None
    max_position_embedding: int | None = None
    rope_scaling: dict | None = None


# @struct.dataclass
# class ModelConfig(C):
#     language_config: LanguageConfig | None = None
#     vision_config: VisionConfig | None = None
#     audio_config: AudioConfig | None = None

