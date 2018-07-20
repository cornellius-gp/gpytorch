#!/usr/bin/env python3

import torch
import gpytorch
from gpytorchopt.additive.additive_structure_gp_model import ExactGPModel
from gpytorchopt.acquisition import ExpectedImprovement
from gpytorchopt.acquisition.strategy.discrete_set import DiscreteSetEvaluationStrategy, DiscreteSetRandomStrategy
from gpytorchopt import BayesianOptimization
import ghalton

# Define the function
func1 = False
func2 = True

if func1:

    def func(x):
        if len(x.shape) == 1:
            return torch.sum(x ** 2)
        else:
            return torch.sum(x ** 2, dim=1)

    # Dimensionality of function to optimize.
    n_dims = 10
    global_opt = torch.zeros(n_dims)
    print(global_opt)
    print("global opt value = %.3f" % (func(global_opt)))

if func2:
    # x should be a torch tensor
    def func(x):
        if len(x.shape) == 1:
            return 0.5 * torch.sum(torch.pow(x, 4) - 16 * torch.pow(x, 2) + 5 * x)
        else:
            return 0.5 * torch.sum(torch.pow(x, 4) - 16 * torch.pow(x, 2) + 5 * x, dim=1)

    # Dimensionality of function to optimize.
    n_dims = 10
    global_opt = -2.903534 * torch.ones(n_dims)
    print(global_opt)
    print("global opt value = %.3f" % (func(global_opt)))


# Define BayesOpt class that uses EI as acquisition function
class BayesianOptimizationFullyDepenedent(BayesianOptimization):
    def __init__(self, model, n_dims, min_bound, max_bound, samples=None, values=None, gpu=False, **kwargs):
        super(BayesianOptimizationFullyDepenedent, self).__init__(
            model, n_dims, min_bound, max_bound, samples, values, gpu, **kwargs
        )
        self._candidate_set = self.cand_generator(n_dims)

    def cand_generator(self, D, candidate_set_size=10000):
        sequencer = ghalton.Halton(D)
        candidate_set = torch.Tensor(sequencer.get(candidate_set_size))
        return candidate_set

    def update_model(self, refit_hyper=True, lr=0.1, optim_steps=300, verbose=False, update_mode="default"):
        initial_kernel = gpytorch.kernels.RBFKernel(log_lengthscale_bounds=(-5, 5))
        likelihood = gpytorch.likelihoods.GaussianLikelihood(log_noise_bounds=(-5, 1))
        self.model = ExactGPModel(self._samples, self._function_values, likelihood, initial_kernel)
        super(BayesianOptimizationFullyDepenedent, self).update_model(
            refit_hyper, lr, optim_steps, verbose, update_mode
        )

    def step(self, function_closure, num_samples=1):
        if self._samples is None:
            random_acqfs = DiscreteSetRandomStrategy(self._candidate_set)
            candidate = random_acqfs.maximize(num_samples, remove_candidate=True)
            self._candidate_set = random_acqfs.candidate_set
        else:
            acquisition_function = ExpectedImprovement(self.model, self.min_value)
            my_acqfs = DiscreteSetEvaluationStrategy(acquisition_function, self._candidate_set)
            candidate = my_acqfs.maximize()
            self._candidate_set = my_acqfs.candidate_set  # new candidate set after the best point is removed

        true_candidate = self.unscale_point(candidate)  # still needs to scale point
        function_value = function_closure(true_candidate)

        self.append_sample(candidate, function_value)
        self.update_model()  # need this ?

        return true_candidate, function_value


initial_kernel = gpytorch.kernels.RBFKernel(log_lengthscale_bounds=(-5, 5))
likelihood = gpytorch.likelihoods.GaussianLikelihood(log_noise_bounds=(-5, 1))  # .cuda()
initial_model = ExactGPModel(
    torch.Tensor([0]), torch.Tensor(), likelihood, initial_kernel
)  # can't unsqueeze empty tensor...

# Number of iterations of Bayesopt to run
num_iters = 200
min_bound = global_opt - 3
max_bound = global_opt + 3
n_candidates = 10000
bo_model = BayesianOptimizationFullyDepenedent(initial_model, n_dims, min_bound, max_bound)

for i in range(num_iters):
    candidate, function_value = bo_model.step(func, 10)
    if candidate.ndimension() == 1 or candidate.shape[0] == 1:
        print(
            "Iteration {it}: objective value = {fval:.3f}, current best = {best:.3f}".format(
                it=i, fval=function_value, best=bo_model.min_value.item()
            )
        )
    else:
        for j in range(candidate.shape[0]):
            print(
                "Iteration {it}: objective value = {fval.3f}, current best = {best.3f}".format(
                    it=i, fval=function_value[j], best=bo_model.min_value.item()
                )
            )
