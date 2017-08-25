import gpytorch
from ..likelihoods import Likelihood, GaussianLikelihood
from ..random_variables import GaussianRandomVariable
from .posterior_models import _ExactGPPosterior, _VariationalGPPosterior
from copy import deepcopy
from torch.autograd import Variable


class Inference(object):
    def __init__(self, gp_model):
        self.gp_model = gp_model
        self.inference_engine = None

        if isinstance(self.gp_model.likelihood, gpytorch.GPModel):
            self.gp_model_inference = Inference(self.gp_model.prior_model)
        else:
            self.gp_model_inference = None

    def restore_from_state_dict(self, state_dict):
        """
        Restores the posterior model (associated with the supplied prior model) from
        a previously-saved state.

        state_dict: the state dict for the posterior model
        """
        # Get training data from previously saved state
        train_x_keys = sorted(key for key in state_dict.keys() if 'train_x' in key)
        train_y_keys = sorted(key for key in state_dict.keys() if 'train_y' in key)
        if len(train_x_keys) == 0:
            raise RuntimeError('Cannot find previously saved train_x data')
        if len(train_y_keys) != 1:
            raise RuntimeError('Previously saved train_y data is in invalid format')

        # Create posterior model using training data
        train_x = tuple(Variable(state_dict[key], volatile=True) for key in train_x_keys)
        train_y = Variable(state_dict[train_y_keys[0]], volatile=True)
        posterior_model = self.run(train_x, train_y)

        # Copy over parameters to posterior model
        posterior_model.load_state_dict(state_dict)
        return posterior_model

    def run_(self, train_x, train_y, **kwargs):
        if isinstance(train_x, Variable):
            train_x = (train_x,)

        if isinstance(self.gp_model, Likelihood):
            raise RuntimeError('Likelihood should not have an inference engine')

        # Replace observation models with posterior versions
        likelihood = self.gp_model.likelihood
        if isinstance(likelihood, GaussianLikelihood):
            output = self.gp_model.forward(*train_x, **kwargs)
            if isinstance(output, GaussianRandomVariable):
                if isinstance(self.gp_model, _ExactGPPosterior):
                    raise RuntimeError('Updating existing GP posteriors is not yet supported.')
                else:
                    self.gp_model = _ExactGPPosterior(self.gp_model, train_x, train_y)
            else:
                raise RuntimeError('Unknown inference type for observation model:\n%s' % repr(self.gp_model))
        else:
            output = self.gp_model.forward(*train_x, **kwargs)
            self.gp_model = _VariationalGPPosterior(self.gp_model, train_x, train_y)

        self.gp_model.eval()
        return self.gp_model

    def run(self, train_x, train_y, **kwargs):
        orig_gp_model = self.gp_model
        self.gp_model = deepcopy(self.gp_model)
        new_gp_model = self.run_(train_x, train_y, **kwargs)
        self.gp_model = orig_gp_model
        return new_gp_model

    def step(self, output):
        return self.inference_engine.step(output)
