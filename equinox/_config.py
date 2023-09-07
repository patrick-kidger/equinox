import os
import warnings
from typing import Literal


EQX_ON_ERROR: Literal["raise", "breakpoint", "nan"] = os.environ.get(  # pyright: ignore
    "EQX_ON_ERROR", "raise"
)
if EQX_ON_ERROR not in ("raise", "breakpoint", "nan"):
    raise ValueError(
        "Unrecognised value for `EQX_ON_ERROR`. Valid values are `EQX_ON_ERROR=raise`, "
        "`EQX_ON_ERROR=breakpoint`, and `EQX_ON_ERROR=nan`."
    )
if EQX_ON_ERROR == "breakpoint":
    warnings.warn(
        "The environment variable `EQX_ON_ERROR=breakpoint` is currently set. Note "
        "that this should only be used for debugging, as it slows down runtime speed."
    )
