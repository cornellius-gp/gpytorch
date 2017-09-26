import torch
from torch.autograd import Variable


class LinearCG(object):
    """
    Implements the linear conjugate gradients method for (approximately) solving systems of the form

        lhs_mat result = rhs_mat

    for positive definite and symmetric matrices lhs_mat. Includes two methods
    for preconditioning: diagonal (Jacobi) preconditioning, which uses
    M=diag(lhs_mat) as a preconditioner, and symmetric successive
    overrelaxation, the form of which is described, e.g., at
    http://netlib.org/linalg/html_templates/node58.html.
    """
    def __init__(self, max_iter=15, tolerance_resid=1e-5, precondition_closure=None):
        self.max_iter = max_iter
        self.tolerance_resid = tolerance_resid
        self.precondition_closure = precondition_closure

    def _diagonal_preconditioner(self, lhs_mat, mat):
        return mat.mul((1 / lhs_mat.diag()).unsqueeze(1).expand_as(mat))

    def _ssor_preconditioner(self, lhs_mat, mat):
        DL = lhs_mat.tril()
        D = lhs_mat.diag()
        upper_part = (1 / D).expand_as(DL).mul(DL.t())
        Minv_times_mat = torch.trtrs(torch.trtrs(mat, DL, upper=False)[0], upper_part)[0]
        return Minv_times_mat

    def solve(self, matmul_closure, rhs, result=None):
        output_dims = rhs.ndimension()
        if output_dims == 1:
            rhs = rhs.unsqueeze(1)

        if isinstance(matmul_closure, Variable) or isinstance(rhs, Variable):
            raise RuntimeError('LinearCG is not intended to operate directly on Variables or be used with autograd.')

        if torch.is_tensor(matmul_closure):
            # If matmul_closure is a tensor, we can use some default preconditioning.
            def default_matmul_closure(tensor):
                return torch.matmul(lhs_mat, tensor)

            lhs_mat = matmul_closure
            matmul_closure = default_matmul_closure

            if lhs_mat.is_cuda:
                self.precondition_closure = lambda mat: mat
                self._reset_precond = True
            else:
                if self.precondition_closure is None:
                    self.precondition_closure = lambda mat: self._ssor_preconditioner(lhs_mat, mat)
                    self._reset_precond = True
                else:
                    self._reset_precond = False

        else:
            # Probably fairly difficult to implement a default preconditioner for an arbitrary mm closure.
            if self.precondition_closure is None:
                self.precondition_closure = lambda mat: mat
            self._reset_precond = True

        # Solve batch
        n, k = rhs.size()

        if result is None:
            result = self.precondition_closure(rhs)

        residuals = rhs - matmul_closure(result)

        # Preconditioner solve is exact in some cases
        rtr = residuals.pow(2).sum(0)
        if not all(rtr.sqrt().squeeze() < self.tolerance_resid):
            z = self.precondition_closure(residuals)
            P = z
            r_dot_zs = residuals.mul(z).sum(0)

            for k in range(min(self.max_iter, n)):
                AP = matmul_closure(P)
                PAPs = AP.mul(P).sum(0)

                alphas = r_dot_zs.div(PAPs + 1e-10)

                result = result + alphas.expand_as(P).mul(P)

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

        if output_dims == 1:
            result = result.squeeze(1)
        return result
