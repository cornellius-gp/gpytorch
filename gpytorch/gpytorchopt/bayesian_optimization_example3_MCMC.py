#!/usr/bin/env python3

import torch
import gpytorch
from gpytorchopt.additive.additive_structure_gp_model import AdditiveStructureGPModel
from gpytorchopt.acquisition.strategy.discrete_set import (
    EnsembleDiscreteSetEvaluationStrategy,
    DiscreteSetRandomStrategy,
)
from gpytorchopt.additive.strategy import DimScanAdditiveStrategy
from gpytorchopt import BayesianOptimization
from gpytorchopt.additive.structure_discovery import MetropolisHastingAdditiveStructureSelector
import ghalton
from gpytorchopt.acquisition import ExpectedImprovement

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


class MyAcquisitionFunctionStrategy:
    def __init__(self, acquisition_function_list, additive_structure_list, candidate_set_factory, models):
        self.acquisition_function_list = acquisition_function_list
        self.additive_structure_list = additive_structure_list
        self.candidate_set_factory = candidate_set_factory
        self.models = models

    def maximize(self):
        cands = []

        for acq_func, additive_structure in zip(self.acquisition_function_list, self.additive_structure_list):
            acq_func_strategy = DimScanAdditiveStrategy(acq_func, self.candidate_set_factory, additive_structure)
            cand = acq_func_strategy.maximize()
            cands.append(cand)

        acq_func_strategy = EnsembleDiscreteSetEvaluationStrategy(
            self.acquisition_function_list, torch.cat(cands), self.models
        )
        candidate = acq_func_strategy.maximize()
        return candidate


# Define BayesOpt class that uses MCMC to discover additive structure
class BayesianOptimizationModelMCMC(BayesianOptimization):
    def update_model(self, refit_hyper=True, lr=0.1, optim_steps=300, verbose=False, update_mode="default"):
        if hasattr(self, "current_kernel"):
            for sub_kernel in self.current_kernel.kernels:
                sub_kernel.initialize(log_lengthscale=0.)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(log_noise_bounds=(-5, 1))
            self.model = AdditiveStructureGPModel(self._samples, self._function_values, likelihood, self.current_kernel)
        super(BayesianOptimizationModelMCMC, self).update_model(refit_hyper, lr, optim_steps, verbose, update_mode)

    def cand_generator(self, D, candidate_set_size=10000):
        sequencer = ghalton.Halton(D)
        candidate_set = torch.Tensor(sequencer.get(candidate_set_size))
        return candidate_set

    def step(self, function_closure, num_samples=1):
        if self._samples is None:
            candidate_set = self.cand_generator(self.n_dims)
            candidate = DiscreteSetRandomStrategy(candidate_set).maximize(num_samples)
        else:
            selector = MetropolisHastingAdditiveStructureSelector(self._samples, self.function_values)
            selector.set_sample(self.model.log_likelihood, self.model, self.model.covar_module)
            models = selector.get_models(num_models=50)
            self.current_kernel = selector.current_kernel
            candidate = MyAcquisitionFunctionStrategy(
                [ExpectedImprovement(model, self.min_value) for model in models],
                [model.additive_structure for model in models],
                self.cand_generator,
                models,
            ).maximize()

        true_candidate = self.unscale_point(candidate)
        function_value = function_closure(true_candidate)
        self.append_sample(candidate, function_value)

        self.update_model()

        return true_candidate, function_value


initial_kernel = gpytorch.kernels.AdditiveKernel(gpytorch.kernels.RBFKernel(active_dims=list(range(n_dims))))
likelihood = gpytorch.likelihoods.GaussianLikelihood(log_noise_bounds=(-5, 1))  # .cuda()
initial_model = AdditiveStructureGPModel(
    torch.Tensor([0]), torch.Tensor(), likelihood, initial_kernel
)  # can't unsqueeze empty tensor...

# Number of iterations of Bayesopt to run
num_iters = 200
min_bound = global_opt - 3
max_bound = global_opt + 3
n_candidates = 10000
bo_model = BayesianOptimizationModelMCMC(initial_model, n_dims, min_bound, max_bound)

for i in range(num_iters):
    candidate, function_value = bo_model.step(func, 2)
    if candidate.ndimension() == 1 or candidate.shape[0] == 1:
        print(
            "Iteration {it}: objective value = {fval:.3f}, current best = {best:.3f}".format(
                it=i, fval=function_value, best=bo_model.min_value.item()
            )
        )
    else:
        for j in range(candidate.shape[0]):
            print(
                "Iteration {it}: objective value = {fval:.3f}, current best = {best:.3f}".format(
                    it=i, fval=function_value[j], best=bo_model.min_value.item()
                )
            )
