import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class FillGrads(Function):
    @staticmethod
    def forward(ctx, loss, *args):
        ctx.save_for_backward(*args[int(len(args) / 2):])
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grads = list(ctx.saved_tensors)[::2] + list(ctx.saved_tensors)[1::2]
        grads = [None] + grads + [None] * len(ctx.saved_tensors)
        return tuple(grads)


def fancy_backward(loss, *args, other_params=None):
    print('op', other_params)
    if other_params is not None:
        other_grads = torch.autograd.grad(loss, other_params, retain_graph=True)
        for param, grad in zip(other_params, other_grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

    mus = args[::2]
    mu_grads = torch.autograd.grad(loss, mus, create_graph=True, retain_graph=True)
    mu_grad_sum = sum(mu_grad.sum() for mu_grad in mu_grads)
    covar_grads = torch.autograd.grad(mu_grad_sum, mus, retain_graph=True)
    covar_grads = [0.5 * covar_grad for covar_grad in covar_grads]
    FillGrads.apply(loss, *args, *mu_grads, *covar_grads).backward()


# a simple test
if __name__ == '__main__':
    import numpy as np

    def loss_fn(z1, z2):
        loss = z1.pow(4.0) + z2.pow(4.0)
        return loss

    mu1 = torch.tensor(0.0, requires_grad=True)
    mu2 = torch.tensor(0.0, requires_grad=True)
    log_sigma1 = torch.tensor(0.0, requires_grad=True)
    log_sigma2 = torch.tensor(0.0, requires_grad=True)

    log_sigma_grads = []

    for k in range(3000):
        covar1 = (2.0 * log_sigma1).exp()
        covar2 = (2.0 * log_sigma2).exp()
        z1 = mu1 + covar1.sqrt() * torch.randn(mu1.shape)
        z2 = mu2 + covar2.sqrt() * torch.randn(mu2.shape)
        loss = loss_fn(z1, z2)
        fancy_backward(loss, mu1, covar1, mu2, covar2)
        log_sigma_grads.append(log_sigma1.grad.item() + log_sigma2.grad.item())
        log_sigma1.grad.zero_(), log_sigma2.grad.zero_()

    print("Mean log_sigma grad: ", np.mean(log_sigma_grads), "  (should be 24ish)")
    assert np.abs(np.mean(log_sigma_grads) - 24.0) < 2.0
