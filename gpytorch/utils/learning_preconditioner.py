import math

import torch
from .. import settings


def learning_low_rank_preconditioner(lzy_tsr, k):
    pd_mat = lzy_tsr.evaluate().detach()

    n = pd_mat.size(-2)

    """
    Crucially, the initialization can NOT be zeros.
    Otherwise first-order methods get stuck because all-zero initializaion is a stationary point.
    """
    L = torch.randn(
        n, k, device=pd_mat.device, dtype=pd_mat.dtype
    ).mul(1. / math.sqrt(n * k)).requires_grad_(True)

    # optimizer = torch.optim.SGD([L], lr=1e-3, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.Adam([L], lr=settings.learning_preconditioner_step_size.value())
    # optimizer = torch.optim.LBFGS([L], lr=settings.learning_preconditioner_step_size.value())

    with torch.enable_grad():
        for i in range(settings.learning_preconditioner_max_iter.value()):
            def objective():
                return L.mm(L.T).sub(pd_mat).square().sum()

            diff = objective()
            diff.backward()

            with torch.no_grad():
                if settings.learning_preconditioner_verbose.on() and i % 50 == 0:
                    print("iter {:d}, fro norm difference {:f}".format(i, diff.sqrt().item()))

                if i < 500:
                    L -= 1. * L.grad / (L.grad.norm() + 1e-10)
                else:
                    L -= 0.1 * L.grad / (L.grad.norm() + 1e-10)

                L.grad.zero_()

            # optimizer.step(objective)
            # optimizer.zero_grad()

    return L.detach()
