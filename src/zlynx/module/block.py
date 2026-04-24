

from flax import nnx
from typing import Tuple, Any, Dict

def create_block(
    num_block: int, 
    module: nnx.Module, 
    *args, rngs: nnx.Rngs, 
    module_args: Tuple[Any, ...] = (),
    module_kwargs: Dict[str, Any] | None = None,
    in_axes: Tuple[Any, ...] = (0,),
    **kwargs
) -> nnx.Module:
    module_kwargs = module_kwargs or {}
    kwarg_keys = tuple(kwargs)
    positional_args = (*args, *kwargs.values())
    num_args = len(args)
    
    @nnx.split_rngs(splits=num_block)
    @nnx.vmap(in_axes=in_axes, out_axes=0)
    def blocks(rngs, *values):
        inner_args = values[:num_args]
        inner_kwargs = dict(zip(kwarg_keys, values[num_args:]))
        return module(
            *module_args,
            *inner_args,
            rngs=rngs,
            **inner_kwargs,
            **module_kwargs,
        )

    return blocks(rngs, *positional_args)

def call_block(
    module: nnx.Module,
    carry: Any,
    *args: Any,
    module_args: Tuple[Any, ...] = (),
    module_kwargs: Dict[str, Any] | None = None,
    in_axes: Tuple[Any, ...] | None = None,
    return_aux: bool = False,
    **kwargs: Any,
) -> Any:
    module_kwargs = module_kwargs or {}
    kwarg_keys = tuple(kwargs)
    positional_args = (*args, *kwargs.values())
    num_args = len(args)

    if in_axes is None:
        in_axes = (0, nnx.Carry, *(0 for _ in positional_args))

    out_axes = (nnx.Carry, 0) if return_aux else nnx.Carry

    @nnx.scan(in_axes=in_axes, out_axes=out_axes)
    def blocks(module, carry, *values):
        inner_args = values[:num_args]
        inner_kwargs = dict(zip(kwarg_keys, values[num_args:]))
        outputs = module(
            carry,
            *module_args,
            *inner_args,
            **inner_kwargs,
            **module_kwargs,
        )

        if not return_aux:
            return outputs[0] if isinstance(outputs, tuple) else outputs

        if not isinstance(outputs, tuple):
            raise TypeError("`return_aux=True` requires the block to return a tuple.")

        next_carry, *aux = outputs
        if len(aux) == 1:
            return next_carry, aux[0]
        return next_carry, tuple(aux)

    return blocks(module, carry, *positional_args)
