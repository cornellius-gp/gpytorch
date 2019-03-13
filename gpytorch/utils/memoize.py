#!/usr/bin/env python3

import functools


def cached(method=None, name=None):
    """A decorator allowing for specifying the name of a cache, allowing it to be modified elsewhere."""
    if method is None:
        return functools.partial(cached, name=name)

    @functools.wraps(method)
    def g(self, *args, **kwargs):
        if not hasattr(self, "_memoize_cache"):
            self._memoize_cache = dict()
        cache_name = name if name is not None else method
        if cache_name not in self._memoize_cache:
            self._memoize_cache[cache_name] = method(self, *args, **kwargs)
        return self._memoize_cache[cache_name]

    return g


def is_cached(self, name):
    """
    Determine if a cached item has been computed
    """
    return hasattr(self, "_memoize_cache") and name in self._memoize_cache.keys()
