from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from gpytorch.priors._compatibility import _bounds_to_prior
from gpytorch.priors import GammaPrior, SmoothedBoxPrior


class TestPriorUtils(unittest.TestCase):
    def test_bounds_to_prior(self):
        prior = GammaPrior(1, 1)
        self.assertEqual(prior, _bounds_to_prior(prior=prior, bounds=None))
        self.assertIsInstance(_bounds_to_prior(prior=None, bounds=(-10, 10)), SmoothedBoxPrior)


if __name__ == "__main__":
    unittest.main()
