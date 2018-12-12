#!/usr/bin/env python3

import functools


def cached(method=None, name=None):
    """A decorator allowing for specifying the name of a cache, allowing it to be modified elsewhere."""
    if method is None:
        return functools.partial(cached, name=name)

    @functools.wraps(method)
    def g(self):
        if not hasattr(self, "__cache"):
            self.__cache = dict()
        cache_name = name if name is not None else method
        if cache_name not in self.__cache:
            self.__cache[cache_name] = method(self)
        return self.__cache[cache_name]

    return g
