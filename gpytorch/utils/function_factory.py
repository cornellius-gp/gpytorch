from torch.autograd import Function
from .lincg import LinearCG


def _default_mm_closure_factor(x):
    return x


def _default_grad_fn(grad_output, rhs_mat):
    return rhs_mat.t().mm(grad_output),


def invmm_factory(mm_closure_factory=_default_mm_closure_factor, grad_fn=_default_grad_fn):
    class Invmm(Function):
        def __init__(self, *args):
            self.args = args

        def forward(self, *args):
            closure_args = self.args + args[:-1]
            rhs_matrix = args[-1]
            res = LinearCG().solve(mm_closure_factory(*closure_args), rhs_matrix)
            if res.ndimension() == 1:
                res.unsqueeze_(1)
            self.save_for_backward(*(list(args) + [res]))
            return res

        def backward(self, grad_output):
            if grad_fn is None:
                raise NotImplementedError

            closure_args = self.args + self.saved_tensors[:-2]
            input_1_t_input_2 = self.saved_tensors[-1]

            closure_arg_grads = [None] * len(closure_args)
            rhs_matrix_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                lhs_matrix_grad = LinearCG().solve(mm_closure_factory(*closure_args), grad_output)
                if lhs_matrix_grad.ndimension() == 1:
                    lhs_matrix_grad.unsqueeze_(1)
                lhs_matrix_grad = lhs_matrix_grad.mul_(-1)
                closure_arg_grads = list(grad_fn(input_1_t_input_2.t(), lhs_matrix_grad.t()))

            # input_2 gradient
            if self.needs_input_grad[1]:
                rhs_matrix_grad = LinearCG().solve(mm_closure_factory(*closure_args), grad_output)

            return tuple(closure_arg_grads + [rhs_matrix_grad])

    return Invmm


def mm_factory(mm_closure_factory=_default_mm_closure_factor, grad_fn=_default_grad_fn):
    class Mm(Function):
        def __init__(self, *args):
            self.args = args

        def forward(self, *args):
            closure_args = self.args + args[:-1]
            rhs_matrix = args[-1]
            res = mm_closure_factory(*closure_args)(rhs_matrix)
            if res.ndimension() == 1:
                res.unsqueeze_(1)
            self.save_for_backward(*(list(args) + [res]))
            return res

        def backward(self, grad_output):
            if grad_fn is None:
                raise NotImplementedError

            closure_args = self.args + self.saved_tensors[:-2]
            input_1_t_input_2 = self.saved_tensors[-1]

            closure_arg_grads = [None] * len(closure_args)
            rhs_matrix_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                closure_arg_grads = list(grad_fn(grad_output, input_1_t_input_2))

            # input_2 gradient
            if self.needs_input_grad[1]:
                rhs_matrix_grad = mm_closure_factory(*closure_args)(grad_output)

            return tuple(closure_arg_grads + [rhs_matrix_grad])

    return Mm
