from .parameter_group import ParameterGroup
from collections import OrderedDict


class MCParameterGroup(ParameterGroup):
    def __init__(self):
        super(MCParameterGroup, self).__init__()
        self._training = False

        self._priors = OrderedDict()
        self._update_buffer = OrderedDict()
        self._posteriors = OrderedDict()

        self._options['num_samples'] = 20

    def has_converged(self, loss_closure):
        return True

    def __getattr__(self, name):
        if self._training and name in self._update_buffer:
            return self._update_buffer[name]
        elif name in self._posteriors.keys():
            return self._posteriors[name]
        elif name in self._priors.keys():
            return self._priors[name]
        else:
            super(MCParameterGroup, self).__getattr__(name)

    def __iter__(self):
        for name in self._priors.keys():
            yield name, getattr(self, name)
