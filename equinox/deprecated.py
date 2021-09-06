import functools as ft
import warnings


def deprecated(*, in_favour_of):
    def decorator(fn):
        msg = f"{fn.__name__} is deprecated in favour of {in_favour_of.__name__}"

        @ft.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(msg)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
