#!/usr/bin/env python3


def length_safe_zip(*args):
    """Python's `zip(...)` with checks to ensure the arguments have
    the same number of elements.

    NOTE: This converts all args that do not define "__len__" to a list.
    """
    args = [a if hasattr(a, "__len__") else list(a) for a in args]
    if len({len(a) for a in args}) > 1:
        raise ValueError(
            "Expected the lengths of all arguments to be equal. Got lengths "
            f"{[len(a) for a in args]} for args {args}. Did you pass in "
            "fewer inputs than expected?"
        )
    return zip(*args)
