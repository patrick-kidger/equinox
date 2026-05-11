import builtins
import jax

from .._ad import filter_custom_jvp


def print(x, msg: str, *args, **kwargs):
    """Prints out a value during the runtime of the program.

    **Arguments:**

    - `x`: will be returned unchanged. This is used to determine where the error check
        happens in the overall computation: it will happen after `x` is computed and
        before the return value is used. `x` can be any PyTree, and it must contain at
        least one array.
    - `msg`, `*args`, `**kwargs`: the string to display. Will be called as
        `print(msg.format(*args, **kwargs))`

    **Returns:**

    The original argument `x` unchanged. **If this return value is unused then the print
    will not be performed.** (It will be removed as part of dead code
    elimination.)

    !!! info

        This function is like `jax.debug.print`, but is 'purely functional': it requires
        an argument `x` to be passed to it and returned from it. This won't matter in
        most programs, but can make `eqx.debug.print` a useful building-block in more
        complicated debugging tools, that e.g. need to respect DCE (dead code
        elimination).
    """
    return _print(x, args, kwargs, msg=msg)


@filter_custom_jvp
def _print(x, args, kwargs, *, msg):
    def _print_impl(_x, _args, _kwargs):
        builtins.print(msg.format(*_args, **_kwargs))
        return _x
    return jax.pure_callback(_print_impl, x, x, args, kwargs)


@_print.def_jvp
def _print_jvp(primals, tangents, *, msg: str):
    x, args, kwargs = primals
    tx, _, _ = tangents
    return _print(x, args, kwargs, msg=msg), tx
