from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gpytorch


class AcquisitionFunction(gpytorch.Module):
    def __init__(self, gp_model):
        super(AcquisitionFunction, self).__init__()
        self.gp_model = gp_model

    def forward(self, candidate_set):
        pass
