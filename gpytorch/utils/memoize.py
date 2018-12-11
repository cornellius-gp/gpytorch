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
