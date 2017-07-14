from torch.autograd import Variable
from .distribution import Distribution
from .random_variables import RandomVariable


class ObservationModel(Distribution):
    def __init__(self, observation_model):
        super(ObservationModel, self).__init__()
        self._parameter_groups = {}
        self.observation_model = observation_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        output = super(ObservationModel, self).__call__(*args, **kwargs)
        if isinstance(output, Variable) or isinstance(output, RandomVariable):
            output = (output,)
        return self.observation_model(*output)
