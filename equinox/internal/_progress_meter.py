import abc
import importlib.util
import threading
from collections.abc import Callable
from typing import Any, cast, Generic, TYPE_CHECKING, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jaxtyping import Array, ArrayLike, Float, Int, PyTree, Real

from .._filters import is_array
from .._module import Module
from .._unvmap import unvmap_all, unvmap_max, unvmap_max_p
from ._nontraceable import nonbatchable


if TYPE_CHECKING:
    _FloatScalarLike = float | Array | np.ndarray
    _IntScalarLike = int | Array | np.ndarray
    _RealScalarLike = bool | int | float | Array | np.ndarray
else:
    _FloatScalarLike = Float[ArrayLike, ""]
    _IntScalarLike = Int[ArrayLike, ""]
    _RealScalarLike = Real[ArrayLike, ""]


_State = TypeVar("_State", bound=PyTree[Array])


class AbstractProgressMeter(Module, Generic[_State]):
    """Progress meters used to indicate how far along an iterative solve is. Typically
    these perform some kind of printout as the solve progresses.
    """

    @abc.abstractmethod
    def init(self) -> _State:
        """Initialises the state for a new progress meter.

        **Arguments:**

        Nothing.

        **Returns:**

        The initial state for the progress meter.
        """

    @abc.abstractmethod
    def step(self, state: _State, progress: _FloatScalarLike) -> _State:
        """Updates the progress meter. Called on every step of an iterative solve.

        **Arguments:**

        - `state`: the state from the previous step.
        - `progress`: how far along the solve is, as a number in `[0, 1]`.

        **Returns:**

        The updated state. In addition, the meter is expected to update as a
        side-effect.
        """

    @abc.abstractmethod
    def close(self, state: _State):
        """Closes the progress meter. Called at the end of an iterative solve.

        **Arguments:**

        - `state`: the final state from the end of the solve.

        *Returns:**

        None.
        """


class NoProgressMeter(AbstractProgressMeter):
    """Indicates that no progress meter should be displayed during the solve."""

    def init(self) -> None:
        return None

    def step(self, state, progress: _FloatScalarLike) -> None:
        del progress
        return state

    def close(self, state):
        del state


NoProgressMeter.__init__.__doc__ = """**Arguments:**

Nothing.
"""


def _unvmap_min(x):  # No `unvmap_min` at the moment.
    # Bind the primitive directly: the typed `unvmap_max` wrapper restricts to
    # `Int[ArrayLike, "..."]`, but the underlying primitive is dtype-generic and
    # `progress` here is a float.
    return -unvmap_max_p.bind(-x)


class _TextProgressMeterState(Module):
    progress: _FloatScalarLike
    meter_idx: _IntScalarLike


class TextProgressMeter(AbstractProgressMeter):
    """A text progress meter, printing out e.g.:
    ```
    0.00%
    2.00%
    5.30%
    ...
    100.00%
    ```
    """

    minimum_increase: _RealScalarLike = 0.02

    @staticmethod
    def _init_bar() -> list[float]:
        print("0.00%")
        return [0.0]

    def init(self) -> _TextProgressMeterState:
        meter_idx = _progress_meter_manager.init(self._init_bar)
        return _TextProgressMeterState(meter_idx=meter_idx, progress=jnp.array(0.0))

    @staticmethod
    def _step_bar(bar: list[float], progress: _FloatScalarLike) -> None:
        if is_array(progress):
            # May not be an array when called with `JAX_DISABLE_JIT=1`
            progress = cast(Array | np.ndarray, progress)
            progress = cast(float, progress.item())
        else:
            progress = cast(float, progress)
        bar[0] = progress
        print(f"{100 * progress:.2f}%")

    def step(
        self, state: _TextProgressMeterState, progress: _FloatScalarLike
    ) -> _TextProgressMeterState:
        # When the surrounding solve is batched, both `state.progress` and `progress`
        # will pick up a batch tracer. (For the former, because the condition for the
        # while-loop-over-steps becomes batched, so necessarily everything in the body
        # of the loop is as well.)
        pred = unvmap_all(
            (progress - state.progress > self.minimum_increase) | (progress == 1)
        )

        # We only print if the progress has increased by at least `minimum_increase` to
        # avoid flooding the user with too many updates.
        next_progress, meter_idx = jax.lax.cond(
            nonbatchable(pred),
            lambda _idx: (
                progress,
                _progress_meter_manager.step(self._step_bar, progress, _idx),
            ),
            lambda _idx: (state.progress, _idx),
            state.meter_idx,
        )

        return _TextProgressMeterState(progress=next_progress, meter_idx=meter_idx)

    @staticmethod
    def _close_bar(bar: list[float]):
        if bar[0] != 1:
            print("100.00%")

    def close(self, state: _TextProgressMeterState):
        _progress_meter_manager.close(self._close_bar, state.meter_idx)


TextProgressMeter.__init__.__doc__ = """**Arguments:**

- `minimum_increase`: the minimum amount the progress has to have increased in order to
    print out a new line. The progress starts at 0 at the beginning of the solve, and
    increases to 1 at the end of the solve. Defaults to `0.02`, so that a new line is
    printed each time the progress increases another 2%.
"""


class _TqdmProgressMeterState(Module):
    meter_idx: _IntScalarLike
    step: _IntScalarLike


class TqdmProgressMeter(AbstractProgressMeter):
    """Uses tqdm to display a progress bar for the solve."""

    refresh_steps: int = 20

    def __check_init__(self):
        if importlib.util.find_spec("tqdm") is None:
            raise ValueError(
                "Cannot use `equinox.internal.TqdmProgressMeter` without `tqdm` "
                "installed. Install it via `pip install tqdm`."
            )

    @staticmethod
    def _init_bar() -> "tqdm.tqdm":  # pyright: ignore  # noqa: F821
        import tqdm  # pyright: ignore

        bar_format = (
            "{percentage:.2f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        return tqdm.tqdm(
            total=100,
            unit="%",
            bar_format=bar_format,
        )

    def init(self) -> _TqdmProgressMeterState:
        meter_idx = _progress_meter_manager.init(self._init_bar)
        return _TqdmProgressMeterState(meter_idx=meter_idx, step=jnp.array(0))

    @staticmethod
    def _step_bar(bar: "tqdm.tqdm", progress: _FloatScalarLike) -> None:  # pyright: ignore  # noqa: F821
        bar.n = round(100 * float(progress), 2)
        bar.update(n=0)
        bar.refresh()

    def step(
        self,
        state: _TqdmProgressMeterState,
        progress: _FloatScalarLike,
    ) -> _TqdmProgressMeterState:
        # Here we update every `refresh_rate` steps in order to limit expensive
        # callbacks.
        # The `unvmap_max` is because batch values for `state.step` start off in sync,
        # and then eventually will freeze their values as that batch element finishes
        # its solve. So take a `max` to get the true number of overall solve steps for
        # the batched system.
        meter_idx = jax.lax.cond(
            nonbatchable(unvmap_max(state.step) % self.refresh_steps == 0),
            lambda _idx: _progress_meter_manager.step(self._step_bar, progress, _idx),
            lambda _idx: _idx,
            state.meter_idx,
        )
        return _TqdmProgressMeterState(meter_idx=meter_idx, step=state.step + 1)

    @staticmethod
    def _close_bar(bar: "tqdm.tqdm"):  # pyright: ignore  # noqa: F821
        bar.n = 100.0
        bar.update(n=0)
        bar.close()

    def close(self, state: _TqdmProgressMeterState):
        _progress_meter_manager.close(self._close_bar, state.meter_idx)


TqdmProgressMeter.__init__.__doc__ = """**Arguments:**

- `refresh_steps`: the number of numerical steps between refreshing the bar. Used to
    limit how frequently the (potentially computationally expensive) bar update is
    performed.
"""


class _ProgressMeterManager:
    """Host-side progress meter manager."""

    def __init__(self):
        self.idx = 0
        self.bars = {}
        # Not sure how important a lock really is, but included just in case.
        self.lock = threading.Lock()

    def init(self, init_bar: Callable[[], Any]) -> _IntScalarLike:
        def _init() -> _IntScalarLike:
            with self.lock:
                bar = init_bar()
                self.idx += 1
                self.bars[self.idx] = bar
                return np.array(self.idx, dtype=jnp.int32)

        # Not `pure_callback` because it's not a deterministic function of its input
        # arguments.
        # Not `debug.callback` because it has a return value.
        meter_idx = io_callback(_init, jax.ShapeDtypeStruct((), jnp.int32))
        return nonbatchable(meter_idx)

    def step(
        self,
        step_bar: Callable[[Any, _FloatScalarLike], None],
        progress: _FloatScalarLike,
        idx: _IntScalarLike,
    ) -> _IntScalarLike:
        # Track the slowest batch element.
        progress = _unvmap_min(progress)

        def _step(_progress, _idx):
            with self.lock:
                try:
                    # This may pick up a spurious batch tracer from a batched condition,
                    # so we need to handle that. We do this by using an `np.unique`.
                    # It should always be the case that `_idx` has precisely one value!
                    bar = self.bars[np.unique(_idx).item()]
                except KeyError:
                    pass  # E.g. the backward pass after a forward pass.
                else:
                    # As above, `_idx` may have a spurious batch tracer. Correspondingly
                    # `_progress` may pick up spurious length-1 batch dimensions from
                    # `vmap_method="expand_dims"` below. Remove them now.
                    step_bar(bar, np.array(_progress).reshape(()))
                # Return the idx to thread the callbacks in the correct order.
                return _idx

        return jax.pure_callback(_step, idx, progress, idx, vmap_method="expand_dims")

    def close(self, close_bar: Callable[[Any], None], idx: _IntScalarLike):
        def _close(_idx):
            with self.lock:
                _idx = _idx.item()
                bar = self.bars[_idx]
                close_bar(bar)
                del self.bars[_idx]

        # Unlike in `step`, we do the `unvmap_max` here. For mysterious reasons this
        # callback does not trigger at all otherwise.
        io_callback(_close, None, unvmap_max(idx))


_progress_meter_manager = _ProgressMeterManager()
