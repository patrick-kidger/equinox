import functools as ft
import warnings


def deprecated(*, in_favour_of):
    if not isinstance(in_favour_of, str):
        in_favour_of = in_favour_of.__name__

    def decorator(fn):
        msg = f"{fn.__name__} is deprecated in favour of {in_favour_of}"

        @ft.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(msg)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
