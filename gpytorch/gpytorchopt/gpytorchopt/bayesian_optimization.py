import torch
from torch import optim
import gpytorch


class BayesianOptimization(object):
    def __init__(self, model, n_dims, min_bound, max_bound, samples=None, values=None, gpu=False, **kwargs):
        self.model = model
        self._samples = samples
        self._function_values = values
        self._dims = n_dims
        self.max_bound = max_bound
        self.min_bound = min_bound
        self._gpu = gpu
        if self._gpu:
            self.model = self.model.cuda()

    def unscale_point(self, x):
        return (self.max_bound - self.min_bound) * x + self.min_bound

    def scale_point(self, x):
        return (x - self.min_bound) / (self.max_bound - self.min_bound)

    def update_model(self, refit_hyper=True, lr=0.1, optim_steps=300, verbose=False, update_mode="default"):
        if self._gpu:
            self.model.set_train_data(self._samples, self._function_values, strict=False).cuda()  # strict ?
        else:
            self.model.set_train_data(self._samples, self._function_values, strict=False)

        if refit_hyper:
            self.model.train()
            self.model.likelihood.train()  # ?? do we need this ??
            # print(len(self._samples))
            optimizer = optim.Adam([{"params": self.model.parameters()}], lr=lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

            for i in range(optim_steps):
                optimizer.zero_grad()
                output = self.model(self._samples)
                loss = -mll(output, self.function_values)
                self.model.log_likelihood = -loss.data[0]  # need this in MCMC, need a way to work around
                if verbose:
                    print(
                        "Iter %d/%d - Loss: %.3f log_outputscale %.3f log_noise: %.3f"
                        % (
                            i + 1,
                            optim_steps,
                            loss.data[0],
                            self.model.log_outputscale.data[0],
                            self.model.likelihood.log_noise.data[0],
                        )
                    )
                loss.backward()
                optimizer.step()
            self.model.eval()

    def append_sample(self, candidate, function_value):
        if candidate.ndimension() == 1:
            candidate = candidate.unsqueeze(0)
        if function_value.ndimension() == 0:
            function_value = function_value.unsqueeze(0)
        if self._samples is None:
            if self._gpu:
                self._samples = candidate.cuda()
                self._function_values = function_value.cuda()
            else:
                self._samples = candidate
                self._function_values = function_value
        else:
            if self._gpu:
                self._samples = torch.cat((self._samples, candidate.cuda()))
                self._function_values = torch.cat((self.function_values, function_value.cuda()))
            else:
                self._samples = torch.cat((self._samples, candidate))
                self._function_values = torch.cat((self.function_values, function_value))

    def step(self, function_closure, num_samples=1):  # what is num_samples for? num_initial?
        raise NotImplementedError

    @property
    def samples(self):
        if self._samples is None:
            return None
        else:
            return self.unscale_point(self._samples)

    @property
    def function_values(self):
        return self._function_values

    @property
    def n_dims(self):
        return self._dims

    @property
    def sorted_samples(self):
        if self._samples is None:
            return None
        else:
            _, inds = self._function_values.sort(0)
            return self.unscale_point(self._samples[inds, :])

    @property
    def sorted_function_values(self):
        if self._function_values is None:
            return None
        else:
            vals, _ = self._function_values.sort()
            return vals

    @property
    def min_sample(self):
        if self._samples is None:
            return None
        else:
            _, ind = self._function_values.min(0)
            return self.unscale_point(self._samples[ind])

    @property
    def min_value(self):
        if self._function_values is None:
            return float("inf")
        else:
            return torch.min(self._function_values)

    @property
    def max_sample(self):
        if self._samples is None:
            return None
        else:
            _, ind = self._function_values.max(0)
            return self.unscale_point(self._samples[ind])

    @property
    def max_value(self):
        if self._function_values is None:
            return float("-inf")
        else:
            return torch.max(self._function_values)
