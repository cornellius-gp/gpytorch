#!/usr/bin/env python3

import math
import os
import random
from abc import ABC, abstractmethod
from itertools import product, combinations
from test._utils import approx_equal

import gpytorch
import torch


class RectangularLazyTensorTestCase(ABC):
    @abstractmethod
    def create_lazy_tensor(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_lazy_tensor(self):
        raise NotImplementedError()

    def setUp(self):
        if hasattr(self.__class__, "seed"):
            seed = self.__class__.seed
            if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
                self.rng_state = torch.get_rng_state()
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                random.seed(seed)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_matmul_vec(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)

        # We skip this test if we're dealing with batch LazyTensors
        # They shouldn't multiply by a vec
        if lazy_tensor.ndimension() > 2:
            return

        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(-1))
        res = lazy_tensor.matmul(test_vector)
        actual = evaluated.matmul(test_vector)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )

    def test_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)
        res = lazy_tensor.matmul(test_vector)
        actual = evaluated.matmul(test_vector)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )

    def test_matmul_matrix_broadcast(self):
        # Right hand size has one more batch dimension
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)
        test_vector = torch.randn(3, *lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)
        res = lazy_tensor.matmul(test_vector)
        actual = evaluated.matmul(test_vector)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )

        if lazy_tensor.ndimension() > 2:
            # Right hand size has one fewer batch dimension
            lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
            lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
            evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)
            test_vector = torch.randn(*lazy_tensor.batch_shape[1:], lazy_tensor.size(-1), 5)
            res = lazy_tensor.matmul(test_vector)
            actual = evaluated.matmul(test_vector)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

            grad = torch.randn_like(res)
            res.backward(gradient=grad)
            actual.backward(gradient=grad)
            for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
                if arg_copy.grad is not None:
                    self.assertLess(
                        ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                    )

            # Right hand size has a singleton dimension
            lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
            lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
            evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)
            test_vector = torch.randn(*lazy_tensor.batch_shape[:-1], 1, lazy_tensor.size(-1), 5)
            res = lazy_tensor.matmul(test_vector)
            actual = evaluated.matmul(test_vector)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

            grad = torch.randn_like(res)
            res.backward(gradient=grad)
            actual.backward(gradient=grad)
            for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
                if arg_copy.grad is not None:
                    self.assertLess(
                        ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                    )

    def test_constant_mul(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)
        self.assertTrue(approx_equal((lazy_tensor * 5).evaluate(), evaluated * 5))

    def test_evaluate(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)
        self.assertTrue(approx_equal(lazy_tensor.evaluate(), evaluated))

    def test_getitem(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        # Non-batch case
        if lazy_tensor.ndimension() == 2:
            res = lazy_tensor[1]
            actual = evaluated[1]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            res = lazy_tensor[0:2].evaluate()
            actual = evaluated[0:2]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            res = lazy_tensor[:, 0:2].evaluate()
            actual = evaluated[:, 0:2]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            res = lazy_tensor[0:2, :].evaluate()
            actual = evaluated[0:2, :]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            res = lazy_tensor[..., 0:2].evaluate()
            actual = evaluated[..., 0:2]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            res = lazy_tensor[0:2, ...].evaluate()
            actual = evaluated[0:2, ...]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            res = lazy_tensor[..., 0:2, 2]
            actual = evaluated[..., 0:2, 2]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            res = lazy_tensor[0:2, ..., 2]
            actual = evaluated[0:2, ..., 2]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)

        # Batch case
        else:
            res = lazy_tensor[1].evaluate()
            actual = evaluated[1]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            res = lazy_tensor[0:2].evaluate()
            actual = evaluated[0:2]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            res = lazy_tensor[:, 0:2].evaluate()
            actual = evaluated[:, 0:2]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)

            for batch_index in product([1, slice(0, 2, None)], repeat=(lazy_tensor.dim() - 2)):
                res = lazy_tensor.__getitem__((*batch_index, slice(0, 1, None), slice(0, 2, None))).evaluate()
                actual = evaluated.__getitem__((*batch_index, slice(0, 1, None), slice(0, 2, None)))
                self.assertEqual(res.shape, actual.shape)
                self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
                res = lazy_tensor.__getitem__((*batch_index, 1, slice(0, 2, None)))
                actual = evaluated.__getitem__((*batch_index, 1, slice(0, 2, None)))
                self.assertEqual(res.shape, actual.shape)
                self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
                res = lazy_tensor.__getitem__((*batch_index, slice(1, None, None), 2))
                actual = evaluated.__getitem__((*batch_index, slice(1, None, None), 2))
                self.assertEqual(res.shape, actual.shape)
                self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)

            # Ellipsis
            res = lazy_tensor.__getitem__((Ellipsis, slice(1, None, None), 2))
            actual = evaluated.__getitem__((Ellipsis, slice(1, None, None), 2))
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            res = lazy_tensor.__getitem__((slice(1, None, None), Ellipsis, 2))
            actual = evaluated.__getitem__((slice(1, None, None), Ellipsis, 2))
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)

    def test_getitem_tensor_index(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        # Non-batch case
        if lazy_tensor.ndimension() == 2:
            index = (torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
            res, actual = lazy_tensor[index], evaluated[index]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            index = (torch.tensor([0, 0, 1, 2]), slice(None, None, None))
            res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]))
            res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            index = (torch.tensor([0, 0, 1, 2]), Ellipsis)
            res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            index = (Ellipsis, torch.tensor([0, 0, 1, 2]))
            res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            index = (Ellipsis, torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
            res, actual = lazy_tensor[index], evaluated[index]
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)

        # Batch case
        else:
            for batch_index in product(
                [torch.tensor([0, 1, 1, 0]), slice(None, None, None)], repeat=(lazy_tensor.dim() - 2)
            ):
                index = (*batch_index, torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1]))
                res, actual = lazy_tensor[index], evaluated[index]
                self.assertEqual(res.shape, actual.shape)
                self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
                index = (*batch_index, torch.tensor([0, 1, 0, 2]), slice(None, None, None))
                res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
                self.assertEqual(res.shape, actual.shape)
                self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
                index = (*batch_index, slice(None, None, None), torch.tensor([0, 1, 2, 1]))
                res, actual = gpytorch.delazify(lazy_tensor[index]), evaluated[index]
                self.assertEqual(res.shape, actual.shape)
                self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
                index = (*batch_index, slice(None, None, None), slice(None, None, None))
                res, actual = lazy_tensor[index].evaluate(), evaluated[index]
                self.assertEqual(res.shape, actual.shape)
                self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)

            # Ellipsis
            res = lazy_tensor.__getitem__((Ellipsis, torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1])))
            actual = evaluated.__getitem__((Ellipsis, torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1])))
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
            res = gpytorch.delazify(
                lazy_tensor.__getitem__((torch.tensor([0, 1, 0, 1]), Ellipsis, torch.tensor([1, 2, 0, 1])))
            )
            actual = evaluated.__getitem__((torch.tensor([0, 1, 0, 1]), Ellipsis, torch.tensor([1, 2, 0, 1])))
            self.assertEqual(res.shape, actual.shape)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)

    def test_permute(self):
        lazy_tensor = self.create_lazy_tensor()
        if lazy_tensor.dim() >= 4:
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)
            dims = torch.randperm(lazy_tensor.dim() - 2).tolist()
            res = lazy_tensor.permute(*dims, -2, -1).evaluate()
            actual = evaluated.permute(*dims, -2, -1)
            self.assertTrue(approx_equal(res, actual))

    def test_quad_form_derivative(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_clone = lazy_tensor.clone().detach_().requires_grad_(True)
        left_vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-2), 2)
        right_vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 2)

        deriv_custom = lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        deriv_auto = gpytorch.lazy.LazyTensor._quad_form_derivative(lazy_tensor_clone, left_vecs, right_vecs)

        for dc, da in zip(deriv_custom, deriv_auto):
            self.assertLess(torch.norm(dc - da), 1e-1)

    def test_sum(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        self.assertTrue(approx_equal(lazy_tensor.sum(-1), evaluated.sum(-1)))
        self.assertTrue(approx_equal(lazy_tensor.sum(-2), evaluated.sum(-2)))
        if lazy_tensor.ndimension() > 2:
            self.assertTrue(approx_equal(lazy_tensor.sum(-3).evaluate(), evaluated.sum(-3)))
        if lazy_tensor.ndimension() > 3:
            self.assertTrue(approx_equal(lazy_tensor.sum(-4).evaluate(), evaluated.sum(-4)))

    def test_transpose_batch(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        if lazy_tensor.dim() >= 4:
            for i, j in combinations(range(lazy_tensor.dim() - 2), 2):
                res = lazy_tensor.transpose(i, j).evaluate()
                actual = evaluated.transpose(i, j)
                self.assertTrue(torch.allclose(res, actual, rtol=1e-4, atol=1e-5))


class LazyTensorTestCase(RectangularLazyTensorTestCase):
    should_test_sample = False
    skip_slq_tests = False

    def test_add_diag(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        other_diag = torch.tensor(1.5)
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + torch.eye(evaluated.size(-1)).view(
            *[1 for _ in range(lazy_tensor.dim() - 2)], evaluated.size(-1), evaluated.size(-1)
        ).repeat(*lazy_tensor.batch_shape, 1, 1).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.tensor([1.5])
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + torch.eye(evaluated.size(-1)).view(
            *[1 for _ in range(lazy_tensor.dim() - 2)], evaluated.size(-1), evaluated.size(-1)
        ).repeat(*lazy_tensor.batch_shape, 1, 1).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.randn(lazy_tensor.size(-1)).pow(2)
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + other_diag.diag().repeat(*lazy_tensor.batch_shape, 1, 1)
        self.assertTrue(approx_equal(res, actual))

        for sizes in product([1, None], repeat=(lazy_tensor.dim() - 2)):
            batch_shape = [lazy_tensor.batch_shape[i] if size is None else size for i, size in enumerate(sizes)]
            other_diag = torch.randn(*batch_shape, lazy_tensor.size(-1)).pow(2)
            res = lazy_tensor.add_diag(other_diag).evaluate()
            actual = evaluated.clone().detach()
            for i in range(other_diag.size(-1)):
                actual[..., i, i] = actual[..., i, i] + other_diag[..., i]
            self.assertTrue(approx_equal(res, actual))

    def test_diag(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        res = lazy_tensor.diag()
        actual = evaluated.diagonal(dim1=-2, dim2=-1)
        actual = actual.view(*lazy_tensor.batch_shape, -1)
        self.assertEqual(res.size(), lazy_tensor.size()[:-1])
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

    def test_inv_matmul_vec(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)

        # We skip this test if we're dealing with batch LazyTensors
        # They shouldn't multiply by a vec
        if lazy_tensor.ndimension() > 2:
            return

        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(-1), requires_grad=True)
        test_vector_copy = test_vector.clone().detach().requires_grad_(True)
        with gpytorch.settings.max_cg_iterations(200):
            res = lazy_tensor.inv_matmul(test_vector)
        actual = evaluated.inverse().matmul(test_vector_copy)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )
        self.assertLess(
            ((test_vector.grad - test_vector_copy.grad).abs() / test_vector.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )

    def test_inv_matmul_vector_with_left(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)

        # We skip this test if we're dealing with batch LazyTensors
        # They shouldn't multiply by a vec
        if lazy_tensor.ndimension() > 2:
            return

        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(-1), requires_grad=True)
        test_left = torch.randn(6, lazy_tensor.size(-1), requires_grad=True)
        test_vector_copy = test_vector.clone().detach().requires_grad_(True)
        test_left_copy = test_left.clone().detach().requires_grad_(True)
        with gpytorch.settings.max_cg_iterations(100):
            res = lazy_tensor.inv_matmul(test_vector, test_left)
        actual = test_left_copy @ evaluated.inverse() @ test_vector_copy
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )
        self.assertLess(
            ((test_left.grad - test_left_copy.grad).abs() / test_left.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )
        self.assertLess(
            ((test_vector.grad - test_vector_copy.grad).abs() / test_vector.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )

    def test_inv_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5, requires_grad=True)
        test_vector_copy = test_vector.clone().detach().requires_grad_(True)
        with gpytorch.settings.max_cg_iterations(100):
            res = lazy_tensor.inv_matmul(test_vector)
        actual = evaluated.inverse().matmul(test_vector_copy)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )
        self.assertLess(
            ((test_vector.grad - test_vector_copy.grad).abs() / test_vector.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )

    def test_inv_matmul_matrix_broadcast(self):
        # Right hand size has one more batch dimension
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)
        test_vector = torch.randn(3, *lazy_tensor.batch_shape, lazy_tensor.size(-1), 5, requires_grad=True)
        test_vector_copy = test_vector.clone().detach().requires_grad_(True)
        with gpytorch.settings.max_cg_iterations(100):
            res = lazy_tensor.inv_matmul(test_vector)
        actual = evaluated.inverse().matmul(test_vector_copy)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )

        if lazy_tensor.ndimension() > 2:
            # Right hand size has one fewer batch dimension
            lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
            lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
            evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)
            test_vector = torch.randn(*lazy_tensor.batch_shape[1:], lazy_tensor.size(-1), 5)
            test_vector_copy = test_vector.clone().detach().requires_grad_(True)
            with gpytorch.settings.max_cg_iterations(100):
                res = lazy_tensor.inv_matmul(test_vector)
            actual = evaluated.inverse().matmul(test_vector_copy)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

            grad = torch.randn_like(res)
            res.backward(gradient=grad)
            actual.backward(gradient=grad)
            for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
                if arg_copy.grad is not None:
                    self.assertLess(
                        ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                    )

            # Right hand size has a singleton dimension
            lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
            lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
            evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)
            test_vector = torch.randn(*lazy_tensor.batch_shape[:-1], 1, lazy_tensor.size(-1), 5, requires_grad=True)
            test_vector_copy = test_vector.clone().detach().requires_grad_(True)
            with gpytorch.settings.max_cg_iterations(100):
                res = lazy_tensor.inv_matmul(test_vector)
            actual = evaluated.inverse().matmul(test_vector_copy)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

            grad = torch.randn_like(res)
            res.backward(gradient=grad)
            actual.backward(gradient=grad)
            for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
                if arg_copy.grad is not None:
                    self.assertLess(
                        ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                    )

    def test_inv_matmul_matrix_with_left(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5, requires_grad=True)
        test_left = torch.randn(*lazy_tensor.batch_shape, 6, lazy_tensor.size(-1), requires_grad=True)
        test_vector_copy = test_vector.clone().detach().requires_grad_(True)
        test_left_copy = test_left.clone().detach().requires_grad_(True)
        with gpytorch.settings.max_cg_iterations(100):
            res = lazy_tensor.inv_matmul(test_vector, test_left)
        actual = test_left_copy @ evaluated.inverse() @ test_vector_copy
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )
        self.assertLess(
            ((test_left.grad - test_left_copy.grad).abs() / test_left.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )
        self.assertLess(
            ((test_vector.grad - test_vector_copy.grad).abs() / test_vector.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )

    def test_inv_quad_logdet(self):
        if not self.__class__.skip_slq_tests:
            # Forward
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)
            flattened_evaluated = evaluated.view(-1, *lazy_tensor.matrix_shape)

            vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 3, requires_grad=True)
            vecs_copy = vecs.clone().detach_().requires_grad_(True)

            with gpytorch.settings.num_trace_samples(128):
                res_inv_quad, res_logdet = lazy_tensor.inv_quad_logdet(inv_quad_rhs=vecs, logdet=True)

            actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum(-2).sum(-1)
            actual_logdet = torch.cat(
                [torch.logdet(flattened_evaluated[i]).unsqueeze(0) for i in range(lazy_tensor.batch_shape.numel())]
            ).view(lazy_tensor.batch_shape)

            diff_invq = (res_inv_quad - actual_inv_quad).abs() / actual_inv_quad.abs().clamp(1, math.inf)
            diff_logdet = (res_logdet - actual_logdet).abs() / actual_logdet.abs().clamp(1, math.inf)
            self.assertLess(diff_invq.max().item(), 0.01)
            self.assertLess(diff_logdet.max().item(), 0.3)

    def test_inv_quad_logdet_no_reduce(self):
        if not self.__class__.skip_slq_tests:
            # Forward
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)
            flattened_evaluated = evaluated.view(-1, *lazy_tensor.matrix_shape)

            vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 3, requires_grad=True)
            vecs_copy = vecs.clone().detach_().requires_grad_(True)

            with gpytorch.settings.num_trace_samples(128):
                res_inv_quad, res_logdet = lazy_tensor.inv_quad_logdet(
                    inv_quad_rhs=vecs, logdet=True, reduce_inv_quad=False
                )

            actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum(-2).sum(-1)
            actual_logdet = torch.cat(
                [torch.logdet(flattened_evaluated[i]).unsqueeze(0) for i in range(lazy_tensor.batch_shape.numel())]
            ).view(lazy_tensor.batch_shape)

            diff_invq = (res_inv_quad.sum(-1) - actual_inv_quad).abs() / actual_inv_quad.abs().clamp(1, math.inf)
            diff_logdet = (res_logdet - actual_logdet).abs() / res_logdet.abs().clamp(1, math.inf)
            self.assertLess(diff_invq.max().item(), 0.01)
            self.assertLess(diff_logdet.max().item(), 0.3)

    def test_prod(self):
        with gpytorch.settings.fast_computations(covar_root_decomposition=False):
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)

            if lazy_tensor.ndimension() > 2:
                self.assertTrue(
                    torch.allclose(lazy_tensor.prod(-3).evaluate(), evaluated.prod(-3), atol=1e-2, rtol=1e-2)
                )
            if lazy_tensor.ndimension() > 3:
                self.assertTrue(
                    torch.allclose(lazy_tensor.prod(-4).evaluate(), evaluated.prod(-4), atol=1e-2, rtol=1e-2)
                )

    def test_root_decomposition(self):
        # Test with Cholesky
        lazy_tensor = self.create_lazy_tensor()
        test_mat = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)
        with gpytorch.settings.max_cholesky_numel(lazy_tensor.matrix_shape.numel() + 1):
            root_approx = lazy_tensor.root_decomposition()
            res = root_approx.matmul(test_mat)
            actual = lazy_tensor.matmul(test_mat)
            self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)

        # Test with Lanczos
        lazy_tensor = self.create_lazy_tensor()
        with gpytorch.settings.max_cholesky_numel(0):
            root_approx = lazy_tensor.root_decomposition()
            res = root_approx.matmul(test_mat)
            actual = lazy_tensor.matmul(test_mat)
            self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)

    def test_root_inv_decomposition(self):
        lazy_tensor = self.create_lazy_tensor()
        root_approx = lazy_tensor.root_inv_decomposition()

        test_mat = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)

        res = root_approx.matmul(test_mat)
        actual = lazy_tensor.inv_matmul(test_mat)
        self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)

    def test_sample(self):
        if self.__class__.should_test_sample:
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)

            samples = lazy_tensor.zero_mean_mvn_samples(50000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
            self.assertLess(((sample_covar - evaluated).abs() / evaluated.abs().clamp(1, math.inf)).max().item(), 3e-1)
