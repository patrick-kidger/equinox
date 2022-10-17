import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu

from ..filters import combine, is_inexact_array, partition
from ..grad import filter_custom_vjp
from ..tree import tree_at


def bounded_while_loop(cond_fun, body_fun, init_val, max_steps):
    if max_steps is None:
        return lax.while_loop(cond_fun, body_fun, init_val)
    return _bounded_while_loop(init_val, cond_fun, body_fun, max_steps)


@filter_custom_vjp
def _bounded_while_loop(init_val, cond_fun, body_fun, max_steps):
    def _cond_fun(carry):
        step, val = carry
        return cond_fun(val) & (step < max_steps)

    def _body_fun(carry):
        step, val = carry
        val = body_fun(val)
        return step + 1, val

    _, final_val = lax.while_loop(_cond_fun, _body_fun, (0, init_val))
    return final_val


def _bounded_while_loop_fwd(init_val, cond_fun, body_fun, max_steps):
    def _vjp_body_fun(val):
        diff_val, nondiff_val = partition(val, is_inexact_array)

        def _diff_body_fun(_diff_val):
            _val = combine(_diff_val, nondiff_val)
            _out = body_fun(_val)
            _diff_out, _static_out = partition(_out, is_inexact_array)
            return _diff_out, _static_out

        diff_out, diff_vjp_fn, static_out = jax.vjp(
            _diff_body_fun, diff_val, has_aux=True
        )
        return combine(diff_out, static_out), diff_vjp_fn

    def expand(x):
        if jnp.issubdtype(x.dtype, jnp.inexact):
            fill_val = jnp.inf
        elif jnp.issubdtype(x.dtype, jnp.integer):
            fill_val = jnp.iinfo(x.dtype).max
        elif jnp.issubdtype(x.dtype, jnp.bool_):
            fill_val = True
        else:
            raise NotImplementedError
        return jnp.full((max_steps,) + x.shape, fill_val, dtype=x.dtype)

    vjp_fn = jax.eval_shape(lambda v: _vjp_body_fun(v)[1], init_val)
    residuals = jtu.tree_map(expand, vjp_fn.args[0].args)
    init_carry = 0, init_val, residuals

    def _cond_fun(carry):
        step, val, _ = carry
        return cond_fun(val) & (step < max_steps)

    def _body_fun(carry):
        step, val, residuals = carry
        val2, vjp_fn = _vjp_body_fun(val)
        residual = vjp_fn.args[0].args
        assign = lambda r, rs: rs.at[step].set(r)
        residuals = jtu.tree_map(assign, residual, residuals)
        assert jtu.tree_structure(val2) == jtu.tree_structure(val)
        assert jtu.tree_structure(residual) == jtu.tree_structure(residuals)
        return step + 1, val2, residuals

    final_carry = _bounded_while_loop(init_carry, _cond_fun, _body_fun, max_steps)
    num_steps, final_val, residuals = final_carry
    vjp_fns = tree_at(lambda v: v.args[0].args, vjp_fn, residuals)
    return final_val, (num_steps, vjp_fns)


def _bounded_while_loop_bwd(
    residuals, grad_final_val, init_val, cond_fun, body_fun, max_steps
):
    del init_val, cond_fun, body_fun
    num_steps, vjp_fns = residuals
    init_carry = num_steps, grad_final_val

    def _cond_fun(carry):
        step, _ = carry
        return step > 0

    def _body_fun(carry):
        step, grad_val = carry
        step = step - 1
        vjp_fn = jtu.tree_map(lambda x: x[step], vjp_fns)
        (grad_val2,) = vjp_fn(grad_val)
        assert jtu.tree_structure(grad_val) == jtu.tree_structure(grad_val2)
        return step, grad_val2

    _, grad_init_val = _bounded_while_loop(init_carry, _cond_fun, _body_fun, max_steps)
    return grad_init_val


_bounded_while_loop.defvjp(_bounded_while_loop_fwd, _bounded_while_loop_bwd)
