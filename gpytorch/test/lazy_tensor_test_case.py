#!/usr/bin/env python3

import math
from abc import abstractmethod
from itertools import combinations, product
from unittest.mock import MagicMock, patch

import torch

import gpytorch

from .base_test_case import BaseTestCase


def _ensure_symmetric_grad(grad):
    """
    A gradient-hook hack to ensure that symmetric matrix gradients are symmetric
    """
    res = torch.add(grad, grad.transpose(-1, -2)).mul(0.5)
    return res


class RectangularLazyTensorTestCase(BaseTestCase):
    @abstractmethod
    def create_lazy_tensor(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_lazy_tensor(self):
        raise NotImplementedError()

    def _test_matmul(self, rhs):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        res = lazy_tensor.matmul(rhs)
        actual = evaluated.matmul(rhs)
        self.assertAllClose(res, actual)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertAllClose(arg.grad, arg_copy.grad, rtol=1e-3)

    def test_matmul_vec(self):
        lazy_tensor = self.create_lazy_tensor()
        rhs = torch.randn(lazy_tensor.size(-1))

        # We skip this test if we're dealing with batch LazyTensors
        # They shouldn't multiply by a vec
        if lazy_tensor.ndimension() > 2:
            return
        else:
            return self._test_matmul(rhs)

    def test_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor()
        rhs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 4)
        return self._test_matmul(rhs)

    def test_matmul_matrix_broadcast(self):
        lazy_tensor = self.create_lazy_tensor()

        # Right hand size has one more batch dimension
        batch_shape = torch.Size((3, *lazy_tensor.batch_shape))
        rhs = torch.randn(*batch_shape, lazy_tensor.size(-1), 4)
        self._test_matmul(rhs)

        if lazy_tensor.ndimension() > 2:
            # Right hand size has one fewer batch dimension
            batch_shape = torch.Size(lazy_tensor.batch_shape[1:])
            rhs = torch.randn(*batch_shape, lazy_tensor.size(-1), 4)
            self._test_matmul(rhs)

            # Right hand size has a singleton dimension
            batch_shape = torch.Size((*lazy_tensor.batch_shape[:-1], 1))
            rhs = torch.randn(*batch_shape, lazy_tensor.size(-1), 4)
            self._test_matmul(rhs)

    def test_constant_mul(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)
        self.assertAllClose((lazy_tensor * 5).evaluate(), evaluated * 5)

    def test_evaluate(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)
        self.assertAllClose(lazy_tensor.evaluate(), evaluated)

    def test_getitem(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        # Non-batch case
        if lazy_tensor.ndimension() == 2:
            res = lazy_tensor[1]
            actual = evaluated[1]
            self.assertAllClose(res, actual)
            res = lazy_tensor[0:2].evaluate()
            actual = evaluated[0:2]
            self.assertAllClose(res, actual)
            res = lazy_tensor[:, 0:2].evaluate()
            actual = evaluated[:, 0:2]
            self.assertAllClose(res, actual)
            res = lazy_tensor[0:2, :].evaluate()
            actual = evaluated[0:2, :]
            self.assertAllClose(res, actual)
            res = lazy_tensor[..., 0:2].evaluate()
            actual = evaluated[..., 0:2]
            self.assertAllClose(res, actual)
            res = lazy_tensor[0:2, ...].evaluate()
            actual = evaluated[0:2, ...]
            self.assertAllClose(res, actual)
            res = lazy_tensor[..., 0:2, 2]
            actual = evaluated[..., 0:2, 2]
            self.assertAllClose(res, actual)
            res = lazy_tensor[0:2, ..., 2]
            actual = evaluated[0:2, ..., 2]
            self.assertAllClose(res, actual)

        # Batch case
        else:
            res = lazy_tensor[1].evaluate()
            actual = evaluated[1]
            self.assertAllClose(res, actual)
            res = lazy_tensor[0:2].evaluate()
            actual = evaluated[0:2]
            self.assertAllClose(res, actual)
            res = lazy_tensor[:, 0:2].evaluate()
            actual = evaluated[:, 0:2]
            self.assertAllClose(res, actual)

            for batch_index in product([1, slice(0, 2, None)], repeat=(lazy_tensor.dim() - 2)):
                res = lazy_tensor.__getitem__((*batch_index, slice(0, 1, None), slice(0, 2, None))).evaluate()
                actual = evaluated.__getitem__((*batch_index, slice(0, 1, None), slice(0, 2, None)))
                self.assertAllClose(res, actual)
                res = lazy_tensor.__getitem__((*batch_index, 1, slice(0, 2, None)))
                actual = evaluated.__getitem__((*batch_index, 1, slice(0, 2, None)))
                self.assertAllClose(res, actual)
                res = lazy_tensor.__getitem__((*batch_index, slice(1, None, None), 2))
                actual = evaluated.__getitem__((*batch_index, slice(1, None, None), 2))
                self.assertAllClose(res, actual)

            # Ellipsis
            res = lazy_tensor.__getitem__((Ellipsis, slice(1, None, None), 2))
            actual = evaluated.__getitem__((Ellipsis, slice(1, None, None), 2))
            self.assertAllClose(res, actual)
            res = lazy_tensor.__getitem__((slice(1, None, None), Ellipsis, 2))
            actual = evaluated.__getitem__((slice(1, None, None), Ellipsis, 2))
            self.assertAllClose(res, actual)

    def test_getitem_tensor_index(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        # Non-batch case
        if lazy_tensor.ndimension() == 2:
            index = (torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
            res, actual = lazy_tensor[index], evaluated[index]
            self.assertAllClose(res, actual)
            index = (torch.tensor([0, 0, 1, 2]), slice(None, None, None))
            res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]))
            res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (torch.tensor([0, 0, 1, 2]), Ellipsis)
            res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (Ellipsis, torch.tensor([0, 0, 1, 2]))
            res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (Ellipsis, torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
            res, actual = lazy_tensor[index], evaluated[index]
            self.assertAllClose(res, actual)

        # Batch case
        else:
            for batch_index in product(
                [torch.tensor([0, 1, 1, 0]), slice(None, None, None)], repeat=(lazy_tensor.dim() - 2)
            ):
                index = (*batch_index, torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1]))
                res, actual = lazy_tensor[index], evaluated[index]
                self.assertAllClose(res, actual)
                index = (*batch_index, torch.tensor([0, 1, 0, 2]), slice(None, None, None))
                res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
                self.assertAllClose(res, actual)
                index = (*batch_index, slice(None, None, None), torch.tensor([0, 1, 2, 1]))
                res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
                self.assertAllClose(res, actual)
                index = (*batch_index, slice(None, None, None), slice(None, None, None))
                res, actual = lazy_tensor[index].evaluate(), evaluated[index]
                self.assertAllClose(res, actual)

            # Ellipsis
            res = lazy_tensor.__getitem__((Ellipsis, torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1])))
            actual = evaluated.__getitem__((Ellipsis, torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1])))
            self.assertAllClose(res, actual)
            res = gpytorch.delazify(
                lazy_tensor.__getitem__((torch.tensor([0, 1, 0, 1]), Ellipsis, torch.tensor([1, 2, 0, 1])))
            )
            actual = evaluated.__getitem__((torch.tensor([0, 1, 0, 1]), Ellipsis, torch.tensor([1, 2, 0, 1])))
            self.assertAllClose(res, actual)

    def test_permute(self):
        lazy_tensor = self.create_lazy_tensor()
        if lazy_tensor.dim() >= 4:
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)
            dims = torch.randperm(lazy_tensor.dim() - 2).tolist()
            res = lazy_tensor.permute(*dims, -2, -1).evaluate()
            actual = evaluated.permute(*dims, -2, -1)
            self.assertAllClose(res, actual)

    def test_quad_form_derivative(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_clone = lazy_tensor.clone().detach_().requires_grad_(True)
        left_vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-2), 2)
        right_vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 2)

        deriv_custom = lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        deriv_auto = gpytorch.lazy.LazyTensor._quad_form_derivative(lazy_tensor_clone, left_vecs, right_vecs)

        for dc, da in zip(deriv_custom, deriv_auto):
            self.assertAllClose(dc, da)

    def test_sum(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        self.assertAllClose(lazy_tensor.sum(-1), evaluated.sum(-1))
        self.assertAllClose(lazy_tensor.sum(-2), evaluated.sum(-2))
        if lazy_tensor.ndimension() > 2:
            self.assertAllClose(lazy_tensor.sum(-3).evaluate(), evaluated.sum(-3))
        if lazy_tensor.ndimension() > 3:
            self.assertAllClose(lazy_tensor.sum(-4).evaluate(), evaluated.sum(-4))

    def test_transpose_batch(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        if lazy_tensor.dim() >= 4:
            for i, j in combinations(range(lazy_tensor.dim() - 2), 2):
                res = lazy_tensor.transpose(i, j).evaluate()
                actual = evaluated.transpose(i, j)
                self.assertAllClose(res, actual, rtol=1e-4, atol=1e-5)


class LazyTensorTestCase(RectangularLazyTensorTestCase):
    should_test_sample = False
    skip_slq_tests = False
    should_call_cg = True
    should_call_lanczos = True

    def _test_inv_matmul(self, rhs, lhs=None, cholesky=False):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)
        evaluated.register_hook(_ensure_symmetric_grad)

        # Create a test right hand side and left hand side
        rhs.requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)
        if lhs is not None:
            lhs.requires_grad_(True)
            lhs_copy = lhs.clone().detach().requires_grad_(True)

        _wrapped_cg = MagicMock(wraps=gpytorch.utils.linear_cg)
        with patch("gpytorch.utils.linear_cg", new=_wrapped_cg) as linear_cg_mock:
            with gpytorch.settings.max_cholesky_size(math.inf if cholesky else 0), gpytorch.settings.cg_tolerance(1e-4):
                # Perform the inv_matmul
                if lhs is not None:
                    res = lazy_tensor.inv_matmul(rhs, lhs)
                    actual = lhs_copy @ evaluated.inverse() @ rhs_copy
                else:
                    res = lazy_tensor.inv_matmul(rhs)
                    actual = evaluated.inverse().matmul(rhs_copy)
                self.assertAllClose(res, actual, rtol=0.02, atol=1e-5)

                # Perform backward pass
                grad = torch.randn_like(res)
                res.backward(gradient=grad)
                actual.backward(gradient=grad)
                for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
                    if arg_copy.grad is not None:
                        self.assertAllClose(arg.grad, arg_copy.grad, rtol=0.03, atol=1e-5)
                self.assertAllClose(rhs.grad, rhs_copy.grad, rtol=0.03, atol=1e-5)
                if lhs is not None:
                    self.assertAllClose(lhs.grad, lhs_copy.grad, rtol=0.03, atol=1e-5)

            # Determine if we've called CG or not
            if not cholesky and self.__class__.should_call_cg:
                self.assertTrue(linear_cg_mock.called)
            else:
                self.assertFalse(linear_cg_mock.called)

    def _test_inv_quad_logdet(self, reduce_inv_quad=True, cholesky=False):
        if not self.__class__.skip_slq_tests:
            # Forward
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)
            flattened_evaluated = evaluated.view(-1, *lazy_tensor.matrix_shape)

            vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 3, requires_grad=True)
            vecs_copy = vecs.clone().detach_().requires_grad_(True)

            _wrapped_cg = MagicMock(wraps=gpytorch.utils.linear_cg)
            with patch("gpytorch.utils.linear_cg", new=_wrapped_cg) as linear_cg_mock:
                with gpytorch.settings.num_trace_samples(256), gpytorch.settings.max_cholesky_size(
                    math.inf if cholesky else 0
                ), gpytorch.settings.cg_tolerance(1e-5):

                    res_inv_quad, res_logdet = lazy_tensor.inv_quad_logdet(
                        inv_quad_rhs=vecs, logdet=True, reduce_inv_quad=reduce_inv_quad
                    )

            actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum(-2)
            if reduce_inv_quad:
                actual_inv_quad = actual_inv_quad.sum(-1)
            actual_logdet = torch.cat(
                [torch.logdet(flattened_evaluated[i]).unsqueeze(0) for i in range(lazy_tensor.batch_shape.numel())]
            ).view(lazy_tensor.batch_shape)

            self.assertAllClose(res_inv_quad, actual_inv_quad, rtol=0.01, atol=0.01)
            self.assertAllClose(res_logdet, actual_logdet, rtol=0.2, atol=0.03)

            if not cholesky and self.__class__.should_call_cg:
                self.assertTrue(linear_cg_mock.called)
            else:
                self.assertFalse(linear_cg_mock.called)

    def test_add_diag(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        other_diag = torch.tensor(1.5)
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + torch.eye(evaluated.size(-1)).view(
            *[1 for _ in range(lazy_tensor.dim() - 2)], evaluated.size(-1), evaluated.size(-1)
        ).repeat(*lazy_tensor.batch_shape, 1, 1).mul(1.5)
        self.assertAllClose(res, actual)

        other_diag = torch.tensor([1.5])
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + torch.eye(evaluated.size(-1)).view(
            *[1 for _ in range(lazy_tensor.dim() - 2)], evaluated.size(-1), evaluated.size(-1)
        ).repeat(*lazy_tensor.batch_shape, 1, 1).mul(1.5)
        self.assertAllClose(res, actual)

        other_diag = torch.randn(lazy_tensor.size(-1)).pow(2)
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + other_diag.diag().repeat(*lazy_tensor.batch_shape, 1, 1)
        self.assertAllClose(res, actual)

        for sizes in product([1, None], repeat=(lazy_tensor.dim() - 2)):
            batch_shape = [lazy_tensor.batch_shape[i] if size is None else size for i, size in enumerate(sizes)]
            other_diag = torch.randn(*batch_shape, lazy_tensor.size(-1)).pow(2)
            res = lazy_tensor.add_diag(other_diag).evaluate()
            actual = evaluated.clone().detach()
            for i in range(other_diag.size(-1)):
                actual[..., i, i] = actual[..., i, i] + other_diag[..., i]
            self.assertAllClose(res, actual, rtol=1e-2, atol=1e-5)

    def test_diag(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        res = lazy_tensor.diag()
        actual = evaluated.diagonal(dim1=-2, dim2=-1)
        actual = actual.view(*lazy_tensor.batch_shape, -1)
        self.assertAllClose(res, actual, rtol=1e-2, atol=1e-5)

    def test_inv_matmul_vector(self, cholesky=False):
        lazy_tensor = self.create_lazy_tensor()
        rhs = torch.randn(lazy_tensor.size(-1))

        # We skip this test if we're dealing with batch LazyTensors
        # They shouldn't multiply by a vec
        if lazy_tensor.ndimension() > 2:
            return
        else:
            return self._test_inv_matmul(rhs)

    def test_inv_matmul_vector_with_left(self, cholesky=False):
        lazy_tensor = self.create_lazy_tensor()
        rhs = torch.randn(lazy_tensor.size(-1))
        lhs = torch.randn(6, lazy_tensor.size(-1))

        # We skip this test if we're dealing with batch LazyTensors
        # They shouldn't multiply by a vec
        if lazy_tensor.ndimension() > 2:
            return
        else:
            return self._test_inv_matmul(rhs, lhs=lhs)

    def test_inv_matmul_vector_with_left_cholesky(self):
        lazy_tensor = self.create_lazy_tensor()
        rhs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)
        lhs = torch.randn(*lazy_tensor.batch_shape, 6, lazy_tensor.size(-1))
        return self._test_inv_matmul(rhs, lhs=lhs, cholesky=True)

    def test_inv_matmul_matrix(self, cholesky=False):
        lazy_tensor = self.create_lazy_tensor()
        rhs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)
        return self._test_inv_matmul(rhs, cholesky=cholesky)

    def test_inv_matmul_matrix_cholesky(self):
        return self.test_inv_matmul_matrix(cholesky=True)

    def test_inv_matmul_matrix_with_left(self):
        lazy_tensor = self.create_lazy_tensor()
        rhs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)
        lhs = torch.randn(*lazy_tensor.batch_shape, 3, lazy_tensor.size(-1))
        return self._test_inv_matmul(rhs, lhs=lhs)

    def test_inv_matmul_matrix_broadcast(self):
        lazy_tensor = self.create_lazy_tensor()

        # Right hand size has one more batch dimension
        batch_shape = torch.Size((3, *lazy_tensor.batch_shape))
        rhs = torch.randn(*batch_shape, lazy_tensor.size(-1), 5)
        self._test_inv_matmul(rhs)

        if lazy_tensor.ndimension() > 2:
            # Right hand size has one fewer batch dimension
            batch_shape = torch.Size(lazy_tensor.batch_shape[1:])
            rhs = torch.randn(*batch_shape, lazy_tensor.size(-1), 5)
            self._test_inv_matmul(rhs)

            # Right hand size has a singleton dimension
            batch_shape = torch.Size((*lazy_tensor.batch_shape[:-1], 1))
            rhs = torch.randn(*batch_shape, lazy_tensor.size(-1), 5)
            self._test_inv_matmul(rhs)

    def test_inv_quad_logdet(self):
        return self._test_inv_quad_logdet(reduce_inv_quad=False, cholesky=False)

    def test_inv_quad_logdet_no_reduce(self):
        return self._test_inv_quad_logdet(reduce_inv_quad=True, cholesky=False)

    def test_inv_quad_logdet_no_reduce_cholesky(self):
        return self._test_inv_quad_logdet(reduce_inv_quad=True, cholesky=True)

    def test_prod(self):
        with gpytorch.settings.fast_computations(covar_root_decomposition=False):
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)

            if lazy_tensor.ndimension() > 2:
                self.assertAllClose(lazy_tensor.prod(-3).evaluate(), evaluated.prod(-3), atol=1e-2, rtol=1e-2)
            if lazy_tensor.ndimension() > 3:
                self.assertAllClose(lazy_tensor.prod(-4).evaluate(), evaluated.prod(-4), atol=1e-2, rtol=1e-2)

    def test_root_decomposition(self, cholesky=False):
        _wrapped_lanczos = MagicMock(wraps=gpytorch.utils.lanczos.lanczos_tridiag)
        with patch("gpytorch.utils.lanczos.lanczos_tridiag", new=_wrapped_lanczos) as lanczos_mock:
            lazy_tensor = self.create_lazy_tensor()
            test_mat = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)
            with gpytorch.settings.max_cholesky_size(math.inf if cholesky else 0):
                root_approx = lazy_tensor.root_decomposition()
                res = root_approx.matmul(test_mat)
                actual = lazy_tensor.matmul(test_mat)
                self.assertAllClose(res, actual, rtol=0.05)

            # Make sure that we're calling the correct function
            if not cholesky and self.__class__.should_call_lanczos:
                self.assertTrue(lanczos_mock.called)
            else:
                self.assertFalse(lanczos_mock.called)

    def test_root_decomposition_cholesky(self):
        return self.test_root_decomposition(cholesky=True)

    def test_root_inv_decomposition(self):
        lazy_tensor = self.create_lazy_tensor()
        root_approx = lazy_tensor.root_inv_decomposition()

        test_mat = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)

        res = root_approx.matmul(test_mat)
        actual = lazy_tensor.inv_matmul(test_mat)
        self.assertAllClose(res, actual, rtol=0.05, atol=0.02)

    def test_sample(self):
        if self.__class__.should_test_sample:
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)

            samples = lazy_tensor.zero_mean_mvn_samples(50000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
            self.assertAllClose(sample_covar, evaluated, rtol=0.3, atol=0.3)
