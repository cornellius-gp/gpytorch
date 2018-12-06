#!/usr/bin/env python3


import functools


def cached(f):
    """A simple caching decorator for instance functions not taking any arguments"""

    @functools.wraps(f)
    def g(self, *args, **kwargs):
        if not hasattr(self, "__cache"):
            self.__cache = f(self)
        return self.__cache

    return g
