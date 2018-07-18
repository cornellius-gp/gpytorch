from math import log
import gpytorch
import random
import torch
from ..additive_structure_gp_model import AdditiveStructureGPModel


def log_fac(n):
    sum = 0
    for i in range(2, n + 1):
        sum += log(i)
    return sum


def log_binomial(n, k):
    return log_fac(n) - log_fac(k) - log_fac(n - k)


class SplitMergeAdditiveStructureProposer:
    def split_kernel_random(self, kernel_list):
        singleton_kernels = [kernel for kernel in kernel_list if len(kernel.active_dims) is 1]
        non_singleton_kernels = [kernel for kernel in kernel_list if len(kernel.active_dims) > 1]

        if len(non_singleton_kernels) == 0:
            return singleton_kernels

        # Get a random kernel from the list
        rand_ind = random.randrange(0, len(non_singleton_kernels))

        kernel_to_split = non_singleton_kernels[rand_ind]

        old_active_dims = kernel_to_split.active_dims

        split_vec = (torch.rand(len(old_active_dims)) < 0.5).type(torch.ByteTensor)

        while torch.sum(split_vec) == 0 or torch.sum(1 - split_vec) == 0:
            split_vec = (torch.rand(len(old_active_dims)) < 0.5).type(torch.ByteTensor)

        kernel1 = gpytorch.kernels.RBFKernel(log_lengthscale_bounds=(-10, 10), active_dims=old_active_dims[split_vec])
        kernel2 = gpytorch.kernels.RBFKernel(
            log_lengthscale_bounds=(-10, 10), active_dims=old_active_dims[1 - split_vec]
        )

        non_singleton_kernels.pop(rand_ind)
        non_singleton_kernels.append(kernel1)
        non_singleton_kernels.append(kernel2)

        new_kernel_list = non_singleton_kernels + singleton_kernels

        # 1/|non_singleton_kernels| * 1/(2^|dims_in_splitted_kernel|)
        # log(1) - log(non_singleton_kernels)
        # log(1) - dims_in_splitted_kernel*log(2)
        proposal_logprob = -log(len(non_singleton_kernels)) + -len(old_active_dims) * log(2)

        reverse_logprob = -log_binomial(len(new_kernel_list), 2)

        return new_kernel_list, proposal_logprob, reverse_logprob

    def merge_kernel_random(self, kernel_list):
        if len(kernel_list) == 1:
            return kernel_list

        rand_ind1 = random.randrange(0, len(kernel_list))
        remaining = list(range(len(kernel_list)))
        remaining.pop(rand_ind1)
        rand_ind2 = random.choice(remaining)

        proposal_logprob = -log_binomial(len(kernel_list), 2)

        old_kernel1 = kernel_list[rand_ind1]
        old_kernel2 = kernel_list[rand_ind2]

        new_active_dims = torch.cat((old_kernel1.active_dims, old_kernel2.active_dims))

        new_kernel = gpytorch.kernels.RBFKernel(log_lengthscale_bounds=(-10, 10), active_dims=new_active_dims)

        kernel_list = [v for i, v in enumerate(kernel_list) if i not in [rand_ind1, rand_ind2]]

        kernel_list.append(new_kernel)

        non_singleton_kernels = [kernel for kernel in kernel_list if len(kernel.active_dims) > 1]
        reverse_logprob = -log(len(non_singleton_kernels)) + -len(new_active_dims) * log(2)

        return kernel_list, proposal_logprob, reverse_logprob

    def propose(self, kernel_list):
        non_singleton_kernels = [kernel for kernel in kernel_list if len(kernel.active_dims) > 1]

        if len(non_singleton_kernels) == 0:
            return self.merge_kernel_random(kernel_list)
        elif len(kernel_list) == 1:
            return self.split_kernel_random(kernel_list)
        elif random.random() < 0.5:
            return self.merge_kernel_random(kernel_list)
        else:
            return self.split_kernel_random(kernel_list)


class MetropolisHastingAdditiveStructureSelector:
    def __init__(self, X, y):
        if X is not None and not isinstance(X, torch.FloatTensor):
            X = torch.Tensor(X).type(torch.FloatTensor)  # .cuda()
        if y is not None and not isinstance(y, torch.FloatTensor):
            y = torch.Tensor(y).type(torch.FloatTensor).squeeze(1)  # .cuda()

        self.X = X
        self.y = y

        self.kernels = []
        self.models = []
        self.trace = []

        self.proposer = SplitMergeAdditiveStructureProposer()
        self.current_model = None
        self.current_kernel = None

    def set_sample(self, trace_value, model, kernel):
        self.trace.append(trace_value)
        self.models.append(model)
        self.current_model = model
        self.current_kernel = kernel

    def get_models(self, num_models):
        for i in range(num_models):
            proposed_sample, logprop, logrev = self.proposer.propose(self.current_kernel.kernels)

            proposed_kernel = gpytorch.kernels.kernel.AdditiveKernel(*proposed_sample)

            likelihood = gpytorch.likelihoods.GaussianLikelihood(log_noise_bounds=(-5, 1))  # .cuda()
            proposed_model = AdditiveStructureGPModel(self.X, self.y, likelihood, proposed_kernel)  # .cuda()

            # train the proposed model
            proposed_model.train()
            likelihood.train()

            optimizer = torch.optim.Adam([{"params": proposed_model.parameters()}], lr=0.1)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, proposed_model)

            training_iter = 300
            for i in range(training_iter):
                optimizer.zero_grad()
                output = proposed_model(self.X)
                loss = -mll(output, self.y)
                proposed_model.log_likelihood = -loss
                loss.backward()
                optimizer.step()

            proposed_loglik = proposed_model.log_likelihood
            top = proposed_loglik + logrev
            bot = self.trace[-1] + logprop
            accept_prob = min(1, torch.exp(top - bot))
            trace_value = proposed_loglik

            accept = random.random() <= accept_prob

            if accept:
                self.set_sample(trace_value, proposed_model, proposed_kernel)
            else:
                self.set_sample(self.trace[-1], self.current_model, self.current_kernel)

        return self.models
