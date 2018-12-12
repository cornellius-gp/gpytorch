#!/usr/bin/env python3

import functools


def cached(f):
    """A simple caching decorator for instance functions not taking any arguments"""

    @functools.wraps(f)
    def g(self):
        if not hasattr(self, "__cache"):
            self.__cache = dict()
        if f not in self.__cache:
            self.__cache[f] = f(self)
        return self.__cache[f]

    return g


def named_cached(name):
    """A decorator allowing for specifying the name of a cache, allowing it to be modified elsewhere."""
    def named_cached_decorator(f):
        @functools.wraps(f)
        def g(self):
            if not hasattr(self, "__cache"):
                self.__cache = dict()
            if name not in self.__cache:
                self.__cache[name] = f(self)
            return self.__cache[name]

        return g

    return named_cached_decorator
