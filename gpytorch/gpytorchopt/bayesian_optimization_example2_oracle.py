import torch
import gpytorch
from gpytorchopt.additive.additive_structure_gp_model import AdditiveStructureGPModel
from gpytorchopt.acquisition_functions import ExpectedImprovement
from gpytorchopt.acquisition_function_strategies.discrete_set import DiscreteSetRandomStrategy
from gpytorchopt.additive.acquisition_function_strategies import AdditiveStrategy, DimScanAdditiveStrategy
from gpytorchopt import BayesianOptimization
import ghalton


# x should be a torch tensor
def stybtang(x):
    if len(x.shape) == 1:
        return 0.5 * torch.sum(torch.pow(x, 4) - 16 * torch.pow(x, 2) + 5 * x)
    else:
        return 0.5 * torch.sum(torch.pow(x, 4) - 16 * torch.pow(x, 2) + 5 * x, dim=1)


# Dimensionality of function to optimize.
n_dims = 10
global_opt = -2.903534 * torch.ones(n_dims)
print(global_opt)
print("global opt value = %.3f" % (stybtang(global_opt)))


# Define BayesOpt class that uses EI as acquisition function
class BayesianOptimizationOracle(BayesianOptimization):
    def cand_generator(self, D, candidate_set_size=10000):
        sequencer = ghalton.Halton(D)
        candidate_set = torch.Tensor(sequencer.get(candidate_set_size))
        return candidate_set

    def update_model(self, refit_hyper=True, lr=0.1, optim_steps=300, verbose=False, update_mode="default"):
        oracle_kernel_list = []
        for i in range(self.n_dims):
            oracle_kernel_list.append(gpytorch.kernels.RBFKernel(log_lengthscale_bounds=(-10, 10), active_dims=[i]))
            oracle_kernel = gpytorch.kernels.AdditiveKernel(*oracle_kernel_list)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(log_noise_bounds=(-5, 1))
            self.model = AdditiveStructureGPModel(self._samples, self._function_values, likelihood, oracle_kernel)
        super(BayesianOptimizationOracle, self).update_model(refit_hyper, lr, optim_steps, verbose, update_mode)

    def step(self, function_closure, num_samples=1):
        if self._samples is None:
            candidate_set = self.cand_generator(self.n_dims)
            candidate = DiscreteSetRandomStrategy(candidate_set).maximize(num_samples)
        else:
            my_acqfs = DimScanAdditiveStrategy(
                ExpectedImprovement(self.model, self.min_value), self.cand_generator, self.model.additive_structure
            )
            candidate = my_acqfs.maximize()

        true_candidate = self.unscale_point(candidate)
        function_value = function_closure(true_candidate)

        self.append_sample(candidate, function_value)
        self.update_model()  # need this ?

        return true_candidate, function_value


oracle_kernel_list = []
for i in range(n_dims):
    oracle_kernel_list.append(gpytorch.kernels.RBFKernel(log_lengthscale_bounds=(-10, 10), active_dims=[i]))
oracle_kernel = gpytorch.kernels.AdditiveKernel(*oracle_kernel_list)
likelihood = gpytorch.likelihoods.GaussianLikelihood(log_noise_bounds=(-5, 1))  # .cuda()
initial_model = AdditiveStructureGPModel(
    torch.Tensor([0]), torch.Tensor(), likelihood, oracle_kernel
)  # can't unsqueeze empty tensor...

# Number of iterations of Bayesopt to run
num_iters = 200
min_bound = global_opt - 3
max_bound = global_opt + 3
n_candidates = 10000
bo_model = BayesianOptimizationOracle(initial_model, n_dims, min_bound, max_bound)

for i in range(num_iters):
    candidate, function_value = bo_model.step(stybtang, 2)
    if candidate.ndimension() == 1 or candidate.shape[0] == 1:
        print("Iteration %d: objective value = %.3f, current best = %.3f" % (i, function_value, bo_model.min_value.item()))
    else:
        for j in range(candidate.shape[0]):
            print("Iteration %d: objective value = %.3f, current best = %.3f" % (
            i, function_value[j], bo_model.min_value.item()))
