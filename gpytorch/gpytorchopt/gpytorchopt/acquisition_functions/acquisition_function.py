import gpytorch


class AcquisitionFunction(gpytorch.Module):
    def __init__(self, gp_model):
        super(AcquisitionFunction, self).__init__()
        self.gp_model = gp_model

    def forward(self, candidate_set):
        # takes in an n*d candidate_set tensor and return an n*1 tensor
        pass
