import torch
import gpytorch
import math
from torch.autograd import Variable


def _identity(x):
    return x


def trace_components(left_matmul_closure, right_matmul_closure, size=None, num_samples=None,
                     estimator_type='mub', tensor_cls=None, use_vars=False, dim_num=None):
    """
    Create components for stochastic trace estimation
    Given two matrices `A` and `B` (represented by closures that define matrix-multiplication
    operations), returns two matrices `A'` and `B'` such that

    ```
    trace( A^T B ) = sum(A' * B') (elementwise operation)
    ```

    If A and B are regular matrices, then this function simply returns A and B,
    since `trace( A^T B ) = sum(A * B)`

    However, if `A` and `B` are more efficiently represented by matrix multiplication closures
    (as is the case if A is a  Toeplitz matrices, if A is the inverse of a PSD matrix, etc.)
    then this will be far more efficient, since `A' and `B'` will have significantly lower dimensionality.

    Args:
        - left_matmul_closure (Tensor or Variable or function(Tensor) -> Tensor or
            function(Variable) -> Variable) defines the left matrix, `A` (n x m), or the
            matrix multiplcation function f(X) = AX

            left_matmul_closure can also be None, which specifies that A is the identity matrix

        - right_matmul_closure (Tensor or Variable or function(Tensor) -> Tensor or
            function(Variable) -> Variable) defines the right matrix, `B` (n x m), or the
            matrix multiplcation function f(X) = BX

            right_matmul_closure can also be None, which specifies that B is the identity matrix

        - size (int) -> the number of columns in the matrices `A` and `B` (i.e. m).
            Only required if left_matmul_closure and right_matmul_closures are not Tensors/Variables

        - num_samples (int) -> the number of random variable samples to stochastically estimate
            trace. Defaults to `gpytorch.functions.num_trace_samples`

        - estimator_type (str) -> Options are 'mub' (Mutually unbiased bases) or
            'hutchinson' (Rademacher random variables). (Default 'mub')

        - use_vars (bool) -> If False, then operations are performed on Tensors.
            If True, then performs operations on Variables. (Default False)
    """
    if torch.is_tensor(left_matmul_closure):
        left_matrix = left_matmul_closure
        size = left_matrix.size(-1)
        tensor_cls = type(left_matrix)
        left_matmul_closure = left_matrix.matmul

    if torch.is_tensor(right_matmul_closure):
        right_matrix = right_matmul_closure
        size = right_matrix.size(-1)
        tensor_cls = type(right_matrix)
        right_matmul_closure = right_matrix.matmul

    if left_matmul_closure is None:
        left_matmul_closure = _identity

    if right_matmul_closure is None:
        right_matmul_closure = _identity

    if size is None:
        raise RuntimeError('Size must be specified, since neither left_matmul_closure nor'
                           ' right_matmul_closure are Tensors/Variables')

    # Default num_samples, tensor_cls
    if num_samples is None:
        num_samples = gpytorch.functions.num_trace_samples

    if tensor_cls is None:
        tensor_cls = torch.Tensor

    # Return A and B if we're using deterministic mode
    if not gpytorch.functions.fastest or size < num_samples:
        eye = tensor_cls(size).fill_(1).diag()
        if use_vars:
            eye = Variable(eye)
        if dim_num is not None:
            eye = eye.expand(dim_num, size, size)
        return left_matmul_closure(eye), right_matmul_closure(eye)

    # Call appropriate estimator
    if estimator_type == 'mub':
        return mubs_trace_components(left_matmul_closure, right_matmul_closure, size, num_samples,
                                     use_vars=use_vars, tensor_cls=tensor_cls, dim_num=dim_num)
    elif estimator_type == 'hutchinson':
        return hutchinson_trace_components(left_matmul_closure, right_matmul_closure, size, num_samples,
                                           use_vars=use_vars, tensor_cls=tensor_cls, dim_num=dim_num)
    else:
        raise RuntimeError('Unknown estimator_type %s' % estimator_type)


def mubs_trace_components(left_matmul_closure, right_matmul_closure, size, num_samples,
                          tensor_cls=torch.Tensor, use_vars=False, dim_num=None):
    r1_coeff = tensor_cls(size)
    torch.arange(0, size, out=r1_coeff)
    r1_coeff.unsqueeze_(1)
    r2_coeff = ((r1_coeff + 1) * (r1_coeff + 2) / 2)

    if dim_num is not None:
        r1 = tensor_cls(num_samples * dim_num).uniform_().mul_(size).floor().type_as(r1_coeff).unsqueeze(1).t()
        r2 = tensor_cls(num_samples * dim_num).uniform_().mul_(size).floor().type_as(r1_coeff).unsqueeze(1).t()
    else:
        r1 = tensor_cls(num_samples).uniform_().mul_(size).floor().type_as(r1_coeff).unsqueeze(1).t()
        r2 = tensor_cls(num_samples).uniform_().mul_(size).floor().type_as(r1_coeff).unsqueeze(1).t()

    two_pi_n = (2 * math.pi) / size
    real_comps = torch.cos(two_pi_n * (r1_coeff.matmul(r1) + r2_coeff.matmul(r2))) / math.sqrt(size)
    imag_comps = torch.sin(two_pi_n * (r1_coeff.matmul(r1) + r2_coeff.matmul(r2))) / math.sqrt(size)

    coeff = math.sqrt(size / num_samples)
    comps = torch.cat([real_comps, imag_comps], 1).mul_(coeff)
    if use_vars:
        comps = Variable(comps)
    if dim_num is not None:
        comps = comps.t().contiguous().view(dim_num, 2 * num_samples, size).transpose(1, 2).contiguous()
    left_res = left_matmul_closure(comps)
    right_res = right_matmul_closure(comps)
    return left_res, right_res


def hutchinson_trace_components(left_matmul_closure, right_matmul_closure, size, num_samples,
                                tensor_cls=torch.Tensor, use_vars=False, dim_num=None):
    coeff = math.sqrt(1. / num_samples)
    if dim_num is not None:
        comps = tensor_cls(size, num_samples * dim_num).bernoulli_().mul_(2).add_(-1).mul_(coeff)
    else:
        comps = tensor_cls(size, num_samples).bernoulli_().mul_(2).add_(-1).mul_(coeff)

    if use_vars:
        comps = Variable(comps)

    if dim_num is not None:
        comps = comps.t().contiguous().view(dim_num, num_samples, size).transpose(1, 2).contiguous()

    left_res = left_matmul_closure(comps)
    right_res = right_matmul_closure(comps)
    return left_res, right_res
