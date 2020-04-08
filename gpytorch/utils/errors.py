#!/usr/bin/env python3


class NanError(RuntimeError):
    pass


class NotPSDError(RuntimeError):
    pass


__all__ = ["NanError", "NotPSDError"]
