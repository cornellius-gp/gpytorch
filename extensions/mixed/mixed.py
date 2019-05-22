import torch
import mixed_cpp


class Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C=None):
        matmul = mixed_cpp.bmm if A.dim() > 2 else mixed_cpp.mm
        C = matmul(A, B, C)
        if any(ctx.needs_input_grad[:2]):
            ctx.save_for_backward(A, B)
        return C

    @staticmethod
    def backward(ctx, gradC):
        A, B = ctx.saved_variables
        matmul = mixed_cpp.bmm if gradC.dim() > 2 else mixed_cpp.mm
        gradA = matmul(gradC, B.transpose(-1, -2), None)
        gradB = matmul(A.transpose(-1, -2), gradC, None)
        #gradA = gradC.matmul(B.transpose(-1, -2))
        #gradB = A.float().transpose(-1, -2).matmul(gradC)
        return gradA.half(), gradB.half(), None


def mm(A, B, C=None):
    return Matmul.apply(A, B, C)


def bmm(A, B, C=None):
    return Matmul.apply(A, B, C)
