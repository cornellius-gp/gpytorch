#!/usr/bin/env python3

import functools
import pickle

from .errors import CachingError


def cached(method=None, name=None, ignore_args=False):
    """A decorator allowing for specifying the name of a cache, allowing it to be modified elsewhere."""
    if ignore_args:
        return _cached_ignore_args(method=method, name=name)
    else:
        return _cached(method=method, name=name)


def add_to_cache(obj, name, val, *args, **kwargs):
    """Add a result to the cache of an object (honoring calling args)."""
    return _add_to_cache(obj, name, val, *args, kwargs_pkl=pickle.dumps(kwargs))


def get_from_cache(obj, name, *args, **kwargs):
    """Get an item from the cache (honoring calling args)."""
    return _get_from_cache(obj, name, *args, kwargs_pkl=pickle.dumps(kwargs))


def pop_from_cache(obj, name, *args, **kwargs):
    """Pop an item from the cache (honoring calling args)."""
    try:
        return obj._memoize_cache.pop((name, args, pickle.dumps(kwargs)))
    except (KeyError, AttributeError):
        raise CachingError("Object does not have item {} stored in cache.".format(name))


def pop_from_cache_ignore_args(obj, name):
    """Pop an item from the cache (honoring calling args)."""
    try:
        return obj._memoize_cache.pop(name)
    except (KeyError, AttributeError):
        raise CachingError("Object does not have item {} stored in cache.".format(name))


def clear_cache_hook(module, *args, **kwargs):
    module._memoize_cache = {}


def _cached(method=None, name=None):
    """A decorator allowing for specifying the name of a cache, allowing it to be modified elsewhere.
    This variant honors the calling args to the decorated function.
    """
    if method is None:
        return functools.partial(_cached, name=name)

    @functools.wraps(method)
    def g(self, *args, **kwargs):
        cache_name = name if name is not None else method
        kwargs_pkl = pickle.dumps(kwargs)
        if not _is_in_cache(self, cache_name, *args, kwargs_pkl=kwargs_pkl):
            return _add_to_cache(self, cache_name, method(self, *args, **kwargs), *args, kwargs_pkl=kwargs_pkl)
        return _get_from_cache(self, cache_name, *args, kwargs_pkl=kwargs_pkl)

    return g


def _cached_ignore_args(method=None, name=None):
    """A decorator allowing for specifying the name of a cache, allowing it to be modified elsewhere.
    This variant ignores the calling args to the decorated function.
    """
    if method is None:
        return functools.partial(_cached_ignore_args, name=name)

    @functools.wraps(method)
    def g(self, *args, **kwargs):
        cache_name = name if name is not None else method
        if not _is_in_cache_ignore_args(self, cache_name):
            return _add_to_cache_ignore_args(self, cache_name, method(self, *args, **kwargs))
        return _get_from_cache_ignore_args(self, cache_name)

    return g


def _add_to_cache(obj, name, val, *args, kwargs_pkl):
    """Add a result to the cache of an object (honoring calling args)."""
    if not hasattr(obj, "_memoize_cache"):
        obj._memoize_cache = {}
    obj._memoize_cache[(name, args, kwargs_pkl)] = val
    return val


def _get_from_cache(obj, name, *args, kwargs_pkl):
    """Get an item from the cache (honoring calling args)."""
    try:
        return obj._memoize_cache[(name, args, kwargs_pkl)]
    except (AttributeError, KeyError):
        raise CachingError("Object does not have item {} stored in cache.".format(name))


def _is_in_cache(obj, name, *args, kwargs_pkl):
    return hasattr(obj, "_memoize_cache") and (name, args, kwargs_pkl) in obj._memoize_cache


def _add_to_cache_ignore_args(obj, name, val):
    """Add a result to the cache of an object (ignoring calling args)."""
    if not hasattr(obj, "_memoize_cache"):
        obj._memoize_cache = {}
    obj._memoize_cache[name] = val
    return val


def _get_from_cache_ignore_args(obj, name):
    """Get an item from the cache (ignoring calling args)."""
    try:
        return obj._memoize_cache[name]
    except (AttributeError, KeyError):
        raise CachingError("Object does not have item {} stored in cache.".format(name))


def _is_in_cache_ignore_args(obj, name):
    return hasattr(obj, "_memoize_cache") and name in obj._memoize_cache


def _is_in_cache_ignore_all_args(obj, name):
    """ checks if item is in cache by name. """
    return hasattr(obj, "_memoize_cache") and name in [x[0] for x in obj._memoize_cache.keys()]
