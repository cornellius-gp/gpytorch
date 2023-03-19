#!/usr/bin/env python3

import torch

from ..module import Module


class GP(Module):
    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model. For a model with `(m)` outputs, a
        `test_batch_shape x q x d`-shaped input to the model in eval mode returns a
        distribution of shape `broadcast(test_batch_shape, model.batch_shape) x q x (m)`.
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(f"{cls_name} does not define batch_shape property")
