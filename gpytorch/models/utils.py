#!/usr/bin/env python3

import torch


def _get_tensor_args(*args):
    for arg in args:
        if torch.is_tensor(arg):
            yield (arg,)
        else:
            yield arg
