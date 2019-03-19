#!/usr/bin/env python3

import functools


def add_to_cache(obj, name, val):
    """Add a result to the cache of an object."""
    if not hasattr(obj, "_memoize_cache"):
        obj._memoize_cache = dict()
    obj._memoize_cache[name] = val
    return obj


def get_from_cache(obj, name):
    """Get an item from the cache."""
    if not is_in_cache(obj, name):
        raise RuntimeError("Object does not have item {} stored in cache.".format(name))
    return obj._memoize_cache[name]


def is_in_cache(obj, name):
    return hasattr(obj, "_memoize_cache") and name in obj._memoize_cache


def cached(method=None, name=None):
    """A decorator allowing for specifying the name of a cache, allowing it to be modified elsewhere."""
    if method is None:
        return functools.partial(cached, name=name)

    @functools.wraps(method)
    def g(self, *args, **kwargs):
        cache_name = name if name is not None else method
        if not is_in_cache(self, cache_name):
            add_to_cache(self, cache_name, method(self, *args, **kwargs))
        return get_from_cache(self, cache_name)

    return g


def is_cached(self, name):
    """
    Determine if a cached item has been computed
    """
    return hasattr(self, "_memoize_cache") and name in self._memoize_cache.keys()
