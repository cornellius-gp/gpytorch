#!/usr/bin/env python3

import types
from typing import Any, Tuple


class AbstractDtype(type):
    """
    A class that mocks out the behavior of jaxtyping.
    This class allows us to use tensor typehints with sizes.
    https://stackoverflow.com/questions/46382170/how-can-i-create-my-own-parameterized-type-in-python-like-optionalt
    """

    def __getitem__(cls, item: Tuple[Any, str]):
        new_cls = types.new_class(
            f"{cls.__name__}_{item[0].__name__}", (cls,), {}, lambda ns: ns.__setitem__("type", item[0])
        )
        return new_cls


class Float(metaclass=AbstractDtype):
    pass


class Long(metaclass=AbstractDtype):
    pass
