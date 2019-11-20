#!/usr/bin/env python3

import os
import random
from abc import ABC

import torch


class BaseTestCase(ABC):
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

    def assertAllClose(self, tensor1, tensor2, rtol=1e-4, atol=1e-5, equal_nan=False):
        if not tensor1.shape == tensor2.shape:
            raise ValueError(f"tensor1 ({tensor1.shape}) and tensor2 ({tensor2.shape}) do not have the same shape.")

        if torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol, equal_nan=equal_nan):
            return True

        if not equal_nan:
            if not torch.equal(tensor1, tensor1):
                raise AssertionError(f"tensor1 ({tensor1.shape}) contains NaNs")
            if not torch.equal(tensor2, tensor2):
                raise AssertionError(f"tensor2 ({tensor2.shape}) contains NaNs")

        rtol_diff = (torch.abs(tensor1 - tensor2) / torch.abs(tensor2)).view(-1)
        rtol_diff = rtol_diff[torch.isfinite(rtol_diff)]
        rtol_max = rtol_diff.max().item()

        atol_diff = (torch.abs(tensor1 - tensor2) - torch.abs(tensor2).mul(rtol)).view(-1)
        atol_diff = atol_diff[torch.isfinite(atol_diff)]
        atol_max = atol_diff.max().item()

        raise AssertionError(
            f"tensor1 ({tensor1.shape}) and tensor2 ({tensor2.shape}) are not close enough. \n"
            f"max rtol: {rtol_max:0.8f}\t\tmax atol: {atol_max:0.8f}"
        )

    def assertEqual(self, item1, item2):
        if torch.is_tensor(item1) and torch.is_tensor(item2):
            if torch.equal(item1, item2):
                return True
            else:
                raise AssertionError(f"{item1} does not equal {item2}.")
        elif torch.is_tensor(item1) or torch.is_tensor(item2):
            raise AssertionError(f"item1 ({type(item1)}) and item2 ({type(item2)}) are not the same type.")
        elif item1 == item2:
            return True
        elif type(item1) != type(item2):
            raise AssertionError(f"item1 ({type(item1)}) and item2 ({type(item2)}) are not the same type.")
        else:
            raise AssertionError(f"tensor1 ({item1}) does not equal tensor2 ({item2}).")
