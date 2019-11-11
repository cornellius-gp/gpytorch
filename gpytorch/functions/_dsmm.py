#!/usr/bin/env python3

from torch.autograd import Function

from ..utils.sparse import bdsmm


class DSMM(Function):
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.sparse = sparse
        return bdsmm(ctx.sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        return None, bdsmm(ctx.sparse.transpose(-1, -2), grad_output)
