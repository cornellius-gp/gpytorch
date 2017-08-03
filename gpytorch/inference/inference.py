from ..likelihoods import Likelihood, GaussianLikelihood
from ..random_variables import GaussianRandomVariable
from .posterior_models import _ExactGPPosterior, _VariationalGPPosterior
from gpytorch import ObservationModel
from copy import deepcopy
from torch.autograd import Variable


class Inference(object):
    def __init__(self, observation_model):
        self.observation_model = observation_model
        self.inference_engine = None

        if isinstance(self.observation_model.observation_model, ObservationModel):
            self.observation_model_inference = Inference(self.observation_model.observation_model)
        else:
            self.observation_model_inference = None

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
        posterior_model = self.run(train_x, train_y, inducing_points=None, optimize=False)

        # Copy over parameters to posterior model
        posterior_model.load_state_dict(state_dict)
        return posterior_model

    def run_(self, train_x, train_y, inducing_points=None, optimize=True, max_inference_steps=20, **kwargs):
        if isinstance(train_x, Variable):
            train_x = (train_x,)

        if inducing_points is None:
            inducing_points = train_x
        else:
            raise RuntimeError('User specified inducing points are not yet supported.')

        if isinstance(inducing_points, Variable):
            inducing_points = (inducing_points, )

        if isinstance(self.observation_model, Likelihood):
            raise RuntimeError('Likelihood should not have an inference engine')

        # Replace observation models with posterior versions
        likelihood = self.observation_model.observation_model
        if isinstance(likelihood, GaussianLikelihood):
            output = self.observation_model.forward(*train_x, **kwargs)
            if len(output) == 2 and isinstance(output[0], GaussianRandomVariable):
                if not isinstance(self.observation_model, _ExactGPPosterior):
                    self.observation_model = _ExactGPPosterior(self.observation_model)

                    def log_likelihood_closure():
                        self.observation_model.zero_grad()
                        output = self.observation_model(*inducing_points)
                        return self.observation_model.marginal_log_likelihood(output, train_y)
                else:
                    raise RuntimeError('Updating existing GP posteriors is not yet supported.')
            else:
                raise RuntimeError('Unknown inference type for observation model:\n%s' % repr(self.observation_model))
        else:
            self.observation_model = _VariationalGPPosterior(self.observation_model, inducing_points)

            def log_likelihood_closure():
                self.observation_model.zero_grad()
                output = self.observation_model.forward(*inducing_points)
                return self.observation_model.marginal_log_likelihood(output, train_y)

        if optimize:
            # Update all parameter groups
            param_groups = list(self.observation_model.parameter_groups())

            has_converged = False
            for i in range(max_inference_steps):
                for param_group in param_groups:
                    param_group.update(log_likelihood_closure)

                has_converged = all([param_group.has_converged(log_likelihood_closure) for param_group in param_groups])
                if has_converged:
                    break

        # Add the data
        self.observation_model.update_data(train_x, train_y)

        return self.observation_model

    def run(self, train_x, train_y, optimize=True, **kwargs):
        orig_observation_model = self.observation_model
        self.observation_model = deepcopy(self.observation_model)
        new_observation_model = self.run_(train_x, train_y, optimize=optimize, **kwargs)
        self.observation_model = orig_observation_model
        return new_observation_model

    def step(self, output):
        return self.inference_engine.step(output)
