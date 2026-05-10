
from flax import nnx

from _transformer import LlamaBlock
from benchmarks._fn import bench, random_input, TestConfig

config = TestConfig(
    vocab_size=320,
    hidden_size=128,
    intermediate_size=192,
)

block = LlamaBlock(config, rngs=nnx.Rngs(42))

inp = random_input(12, config.hidden_size)

print(block(inp))