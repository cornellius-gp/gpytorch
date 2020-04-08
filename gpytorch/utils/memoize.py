#!/usr/bin/env python3

import functools
import pickle


def add_to_cache(obj, name, val, *args, **kwargs):
    """Add a result to the cache of an object."""
    return _add_to_cache(obj, name, val, *args, kwargs_pkl=pickle.dumps(kwargs))


def _add_to_cache(obj, name, val, *args, kwargs_pkl):
    """Add a result to the cache of an object."""
    if not hasattr(obj, "_memoize_cache"):
        obj._memoize_cache = {}
    obj._memoize_cache[(name, args, kwargs_pkl)] = val
    return obj


def _get_from_cache(obj, name, *args, kwargs_pkl):
    """Get an item from the cache."""
    if not _is_in_cache(obj, name, *args, kwargs_pkl=kwargs_pkl):
        raise RuntimeError("Object does not have item {} stored in cache.".format(name))
    return obj._memoize_cache[(name, args, kwargs_pkl)]


def _is_in_cache(obj, name, *args, kwargs_pkl):
    return hasattr(obj, "_memoize_cache") and (name, args, kwargs_pkl) in obj._memoize_cache


def cached(method=None, name=None):
    """A decorator allowing for specifying the name of a cache, allowing it to be modified elsewhere."""
    if method is None:
        return functools.partial(cached, name=name)

    @functools.wraps(method)
    def g(self, *args, **kwargs):
        cache_name = name if name is not None else method
        kwargs_pkl = pickle.dumps(kwargs)
        if not _is_in_cache(self, cache_name, *args, kwargs_pkl=kwargs_pkl):
            _add_to_cache(self, cache_name, method(self, *args, **kwargs), *args, kwargs_pkl=kwargs_pkl)
        return _get_from_cache(self, cache_name, *args, kwargs_pkl=kwargs_pkl)

    return g
