import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.linear_cg import linear_cg
from gpytorch.utils.linear_log_re_cg import linear_log_cg_re
from gpytorch.settings import num_trace_samples
from gpytorch.settings import max_cg_iterations
from gpytorch.settings import cg_tolerance


class HalfEMLL(ExactMarginalLogLikelihood):
    def __init__(self, likelihood, model, all_half_mvms=False):
        super().__init__(likelihood=likelihood, model=model)
        self.all_half_mvms = all_half_mvms
        self.solver_fn = linear_cg
        self.x0 = None

    def update_x0(self, full_rhs):
        x0 = torch.zeros_like(full_rhs)
        return x0

    def forward(self, function_dist, target, *params):
        function_dist = self.likelihood(function_dist)

        full_rhs, probe_vectors = get_rhs_and_probes(rhs=target - function_dist.mean)
        kxx = function_dist.lazy_covariance_matrix.evaluate_kernel()
        precond, *_ = kxx._preconditioner()
        kxx_h = kxx.half()
        scaling = kxx.shape[-1] ** 0.5

        def compute_half_matmul(x):
            output = kxx_h.matmul(x.half() / scaling).float() * scaling
            return torch.nan_to_num(output, nan=1e-6)

        forwards_matmul = compute_half_matmul if self.all_half_mvms else kxx.matmul

        x0 = self.update_x0(full_rhs)
        with torch.no_grad():
            solve = self.solver_fn(
                matmul_closure=compute_half_matmul,
                rhs=full_rhs,
                initial_guess=x0,
                max_iter=max_cg_iterations.value(),
                tolerance=cg_tolerance.value(),
                preconditioner=precond,
                max_tridiag_iter=0,
            )
            solve = torch.nan_to_num(solve)
        self.x0 = solve.clone()

        pseudo_loss = self.compute_pseudo_loss(
            forwards_matmul=forwards_matmul,
            solve=solve,
            probe_vectors=probe_vectors,
            function_dist=function_dist,
            params=params,
        )
        return pseudo_loss

    def compute_pseudo_loss(
        self, forwards_matmul, solve, probe_vectors, function_dist, params
    ):
        data_solve = solve[..., 0].unsqueeze(-1).contiguous()
        data_term = (-data_solve * forwards_matmul(data_solve).float()).sum(-2) / 2
        logdet_term = (
            (solve[..., 1:] * forwards_matmul(probe_vectors).float()).sum(-2)
            / (2 * probe_vectors.shape[-1])
        )
        res = -data_term - logdet_term.sum(-1)
        res = self._add_other_terms(res, params)
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)


def get_rhs_and_probes(rhs):
    num_random_probes = num_trace_samples.value()
    probe_vectors = torch.randn(
        rhs.shape[-1], num_random_probes, device=rhs.device, dtype=rhs.dtype
    ).contiguous()
    full_rhs = torch.cat((rhs.unsqueeze(-1), probe_vectors), -1)
    return full_rhs, probe_vectors


class ReHalfEMLL(HalfEMLL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solver_fn = linear_log_cg_re


class WarmHalfEMLL(HalfEMLL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solver_fn = linear_log_cg_re

    def update_x0(self, full_rhs):
        x0 = update_initial_guess(self.x0, full_rhs)
        return x0


class WarmReHalfEMLL(ReHalfEMLL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_x0(self, full_rhs):
        x0 = update_initial_guess(self.x0, full_rhs)
        return x0


def update_initial_guess(x0, v):
    x0 = torch.zeros_like(v) if x0 is None else x0.clone()
    x0 = torch.nan_to_num(x0)
    x0 = torch.clip(x0, min=-1.0e4, max=1.0e4)
    return x0
