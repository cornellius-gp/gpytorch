#!/usr/bin/env python3

import unittest
import warnings

import linear_operator
import torch

import gpytorch
from gpytorch.test.base_test_case import BaseTestCase

lazy_tensor_map = {
    "AddedDiagLazyTensor": linear_operator.operators.AddedDiagLinearOperator,
    "BatchRepeatLazyTensor": linear_operator.operators.BatchRepeatLinearOperator,
    "BlockDiagLazyTensor": linear_operator.operators.BlockDiagLinearOperator,
    "BlockInterleavedLazyTensor": linear_operator.operators.BlockInterleavedLinearOperator,
    "BlockLazyTensor": linear_operator.operators.BlockLinearOperator,
    "CatLazyTensor": linear_operator.operators.CatLinearOperator,
    "CholLazyTensor": linear_operator.operators.CholLinearOperator,
    "ConstantMulLazyTensor": linear_operator.operators.ConstantMulLinearOperator,
    "ConstantDiagLazyTensor": linear_operator.operators.ConstantDiagLinearOperator,
    "DiagLazyTensor": linear_operator.operators.DiagLinearOperator,
    "IdentityLazyTensor": linear_operator.operators.IdentityLinearOperator,
    "InterpolatedLazyTensor": linear_operator.operators.InterpolatedLinearOperator,
    "KeOpsLazyTensor": linear_operator.operators.KeOpsLinearOperator,
    "KroneckerProductAddedDiagLazyTensor": linear_operator.operators.KroneckerProductAddedDiagLinearOperator,
    "KroneckerProductDiagLazyTensor": linear_operator.operators.KroneckerProductDiagLinearOperator,
    "KroneckerProductLazyTensor": linear_operator.operators.KroneckerProductLinearOperator,
    "KroneckerProductTriangularLazyTensor": linear_operator.operators.KroneckerProductTriangularLinearOperator,
    "LazyTensor": linear_operator.operators.LinearOperator,
    "LowRankRootAddedDiagLazyTensor": linear_operator.operators.LowRankRootAddedDiagLinearOperator,
    "LowRankRootLazyTensor": linear_operator.operators.LowRankRootLinearOperator,
    "MatmulLazyTensor": linear_operator.operators.MatmulLinearOperator,
    "MulLazyTensor": linear_operator.operators.MulLinearOperator,
    "NonLazyTensor": linear_operator.operators.DenseLinearOperator,
    "PsdSumLazyTensor": linear_operator.operators.PsdSumLinearOperator,
    "RootLazyTensor": linear_operator.operators.RootLinearOperator,
    "SumBatchLazyTensor": linear_operator.operators.SumBatchLinearOperator,
    "SumKroneckerLazyTensor": linear_operator.operators.SumKroneckerLinearOperator,
    "SumLazyTensor": linear_operator.operators.SumLinearOperator,
    "ToeplitzLazyTensor": linear_operator.operators.ToeplitzLinearOperator,
    "TriangularLazyTensor": linear_operator.operators.TriangularLinearOperator,
    "ZeroLazyTensor": linear_operator.operators.ZeroLinearOperator,
}


class LazyTensorImport(BaseTestCase, unittest.TestCase):
    def test_deprecated_imports(self):
        for lt_class_name, linear_op_class in lazy_tensor_map.items():
            with warnings.catch_warnings(record=True) as ws:
                lt_class = getattr(gpytorch.lazy, lt_class_name)
                self.assertEqual(lt_class, linear_op_class)
                self.assertTrue(len(ws) == 1)
                self.assertTrue(issubclass(ws[0].category, DeprecationWarning))

    def test_deprecated_methods(self):
        with warnings.catch_warnings(record=True) as ws:
            mat = torch.randn(5, 5)
            mat = mat @ mat.mT
            nlt = gpytorch.lazy.NonLazyTensor(mat)
            dlo = linear_operator.operators.DenseLinearOperator(mat)
            self.assertTrue(len(ws) == 1)
            self.assertTrue(issubclass(ws[0].category, DeprecationWarning))

        with warnings.catch_warnings(record=True) as ws:
            diag = torch.randn(5)
            self.assertAllClose(nlt.add_diag(diag).to_dense(), dlo.add_diagonal(diag).to_dense())
            self.assertTrue(len(ws) == 1)
            self.assertTrue(issubclass(ws[0].category, DeprecationWarning))

        with warnings.catch_warnings(record=True) as ws:
            self.assertAllClose(nlt.diag(), dlo.diag())
            self.assertTrue(len(ws) == 1)
            self.assertTrue(issubclass(ws[0].category, DeprecationWarning))

        with warnings.catch_warnings(record=True) as ws:
            self.assertAllClose(nlt.evaluate(), dlo.to_dense())
            self.assertTrue(len(ws) == 1)
            self.assertTrue(issubclass(ws[0].category, DeprecationWarning))

        with warnings.catch_warnings(record=True) as ws:
            rhs = torch.randn(5, 2)
            self.assertAllClose(nlt.inv_matmul(rhs), dlo.solve(rhs))
            self.assertTrue(len(ws) == 1)
            self.assertTrue(issubclass(ws[0].category, DeprecationWarning))

        with warnings.catch_warnings(record=True) as ws:
            self.assertAllClose(nlt.symeig(eigenvectors=True)[0], torch.linalg.eigh(dlo)[0])
            self.assertAllClose(nlt.symeig(eigenvectors=True)[1].to_dense(), torch.linalg.eigh(dlo)[1].to_dense())
            self.assertTrue(len(ws) == 1)
            self.assertTrue(issubclass(ws[0].category, DeprecationWarning))

        with warnings.catch_warnings(record=True) as ws:
            self.assertAllClose(nlt.symeig(eigenvectors=False)[0], torch.linalg.eigvalsh(dlo))
            self.assertTrue(len(ws) == 1)
            self.assertTrue(issubclass(ws[0].category, DeprecationWarning))

        with warnings.catch_warnings(record=True) as ws:
            a = torch.randn(5, 2)
            b = torch.randn(5, 2)
            self.assertAllClose(nlt._quad_form_derivative(a, b)[0], dlo._bilinear_derivative(a, b)[0])
            self.assertTrue(len(ws) == 1)
            self.assertTrue(issubclass(ws[0].category, DeprecationWarning))
