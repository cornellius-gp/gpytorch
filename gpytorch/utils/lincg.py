import math
import torch
import gpytorch
from torch.autograd import Variable

import pdb

class LinearCG(object):
    """
    Implements the linear conjugate gradients method for (approximately) solving systems of the form

        Ax = b

    for positive definite and symmetric matrices A. Includes two methods for preconditioning: diagonal (Jacobi)
    preconditioning, which uses M=diag(A) as a preconditioner, and symmetric successive overrelaxation, the form of
    which is described, e.g., at http://netlib.org/linalg/html_templates/node58.html.
    """
    def __init__(self, max_iter=15, tolerance_resid=1e-5, precondition_closure=None):
        self.max_iter = max_iter
        self.tolerance_resid = tolerance_resid
        self.precondition_closure = precondition_closure

    def _diagonal_preconditioner(self, A, v):
        if v.ndimension() > 1:
            return v.mul((1 / A.diag()).unsqueeze(1).expand_as(v))
        else:
            return v.mul((1 / A.diag()).expand_as(v))

    def _ssor_preconditioner(self, A, v):
        DL = A.tril()
        D = A.diag()
        upper_part = (1 / D).expand_as(DL).mul(DL.t())
        Minv_times_v = torch.trtrs(torch.trtrs(v, DL, upper=False)[0], upper_part)[0].squeeze()
        return Minv_times_v

    def solve(self, A, b, x=None):
        b = b.squeeze()

        if isinstance(A, Variable) or isinstance(b, Variable):
            raise RuntimeError('LinearCG is not intended to operate directly on Variables or be used with autograd.')

        if isinstance(A, torch.Tensor):
            # If A is a tensor, we can use some default preconditioning.
            if self.precondition_closure is None:
                self.precondition_closure = lambda v: self._ssor_preconditioner(A, v)
                self._reset_precond = True
            else:
                self._reset_precond = False
            mv_closure = lambda v: A.mv(v)
        else:
            # Probably fairly difficult to implement a default preconditioner for an arbitrary mv closure.
            if self.precondition_closure is None:
                self.precondition_closure = lambda v: v
            mv_closure = A
            self._reset_precond = True

        if b.ndimension() > 1:
            return self._solve_batch(A, b, x)

        if x is None:
            x = self.precondition_closure(b)

        residual = b - mv_closure(x)

        # Preconditioner solve is exact in some cases
        rtr = residual.dot(residual)
        if math.sqrt(rtr) < self.tolerance_resid:
            return x

        z = self.precondition_closure(residual)
        p = z

        r_dot_z = residual.dot(z)

        for k in range(self.max_iter):
            Ap = mv_closure(p)
            alpha = r_dot_z / (p.dot(Ap) + 1e-10)

            x = x + alpha * p
            residual = residual - alpha * Ap

            rtr = residual.dot(residual)
            if math.sqrt(rtr) < self.tolerance_resid:
                break

            z = self.precondition_closure(residual)

            new_r_dot_z = residual.dot(z)

            beta = new_r_dot_z / (r_dot_z + 1e-10)

            p = z + beta * p
            r_dot_z = new_r_dot_z

        if self._reset_precond:
            self.precondition_closure = None

        return x

    def _solve_batch(self, A, B, X=None):
        n, k = B.size()
        if isinstance(A, Variable) or isinstance(B, Variable):
            raise RuntimeError('LinearCG is not intended to operate directly on Variables or be used with autograd.')

        if isinstance(A, torch.Tensor):
            mm_closure = lambda M: A.mm(M)
        else:
            mm_closure = A

        if X is None:
            X = self.precondition_closure(B)

        residuals = B - mm_closure(X)

        # Preconditioner solve is exact in some cases
        rtr = residuals.pow(2).sum(0)
        if all(rtr.sqrt().squeeze() < self.tolerance_resid):
            return X

        z = self.precondition_closure(residuals)
        P = z
        r_dot_zs = residuals.mul(z).sum(0)

        for k in range(min(self.max_iter, n)):
            AP = mm_closure(P)
            PAPs = AP.mul(P).sum(0)

            alphas = r_dot_zs.div(PAPs + 1e-10)

            X = X + alphas.expand_as(P).mul(P)

            residuals = residuals - alphas.expand_as(AP).mul(AP)

            r_sq_news = residuals.pow(2).sum(0)

            if all(r_sq_news.sqrt().squeeze() < self.tolerance_resid):
                break

            z = self.precondition_closure(residuals)
            new_r_dot_zs = residuals.mul(z).sum(0)

            betas = new_r_dot_zs.div(r_dot_zs + 1e-10)

            P = z + betas.expand_as(P).mul(P)
            r_dot_zs = new_r_dot_zs

        if self._reset_precond:
            self.precondition_closure = None

        return X
