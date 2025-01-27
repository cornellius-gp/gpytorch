#!/usr/bin/env python3

from torch import Tensor

from ..module import Module


class GP(Module):
    def apply_input_transforms(self, X: Tensor, is_training_input: bool) -> Tensor:
        input_transform = getattr(self, "input_transform", None)
        if input_transform is not None:
            return input_transform(X=X, is_training_input=is_training_input)
        else:
            return X
