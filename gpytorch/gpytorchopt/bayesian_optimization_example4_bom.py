import torch
import gpytorch
from gpytorchopt.additive.additive_structure_gp_model import ExactGPModel
from gpytorchopt.acquisition_functions import ExpectedImprovement
from gpytorchopt.acquisition_function_strategies.discrete_set import DiscreteSetRandomStrategy
from gpytorchopt.additive.acquisition_function_strategies import AdditiveStrategy, DimScanAdditiveStrategy
from gpytorchopt import BayesianOptimization
from gpytorchopt.additive.structure_discovery import BagofModelsAdditiveStructureSelector
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
class BayesianOptimizationRandom(BayesianOptimization):
    def cand_generator(self, D, candidate_set_size=10000):
        sequencer = ghalton.Halton(D)
        candidate_set = torch.Tensor(sequencer.get(candidate_set_size))
        return candidate_set

    def step(self, function_closure, num_samples=1):
        if self.samples is None:
            candidate_set = self.cand_generator(self.n_dims)
            candidate = DiscreteSetRandomStrategy(candidate_set).maximize(num_samples)
        else:
            selector = BagofModelsAdditiveStructureSelector(self._samples, self._function_values, self.n_dims)
            self.model = selector.get_models(num_models=50)

            acq_func_strategy = DimScanAdditiveStrategy(
                ExpectedImprovement(self.model, self.min_value), self.cand_generator, self.model.additive_structure
            )
            # candidate_set = self.scale_point(self.min_sample).repeat(10000, 1)
            candidate = acq_func_strategy.maximize()

        true_candidate = self.unscale_point(candidate)
        function_value = function_closure(true_candidate)
        self.append_sample(candidate, function_value)

        return true_candidate, function_value


initial_kernel = gpytorch.kernels.AdditiveKernel(gpytorch.kernels.RBFKernel(active_dims=list(range(n_dims))))
likelihood = gpytorch.likelihoods.GaussianLikelihood(log_noise_bounds=(-5, 1))  # .cuda()
initial_model = ExactGPModel(
    torch.Tensor([0]), torch.Tensor(), likelihood, initial_kernel
)  # can't unsqueeze empty tensor...

# Number of iterations of Bayesopt to run
num_iters = 200
min_bound = global_opt - 3
max_bound = global_opt + 3
n_candidates = 10000
bo_model = BayesianOptimizationRandom(initial_model, n_dims, min_bound, max_bound)

for i in range(num_iters):
    candidate, function_value = bo_model.step(func, 2)
    if candidate.ndimension() == 1 or candidate.shape[0] == 1:
        print("Iteration %d: objective value = %.3f, current best = %.3f" % (i, function_value, bo_model.min_value.item()))
    else:
        for j in range(candidate.shape[0]):
            print("Iteration %d: objective value = %.3f, current best = %.3f" % (
            i, function_value[j], bo_model.min_value.item()))
