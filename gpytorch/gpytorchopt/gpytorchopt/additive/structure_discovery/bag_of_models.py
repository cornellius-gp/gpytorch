from math import log
import gpytorch
import random
import numpy as np
import torch
from ..additive_structure_gp_model import AdditiveStructureGPModel


class ShiftDimAdditiveStructureProposer:
    def __init__(self):
        self.num_dims = None

    def propose(self, kernel_list):
        if self.num_dims is None:
            self.num_dims = sum([len(k.active_dims) for k in kernel_list])

        rand_dim = random.randrange(0, self.num_dims)
        old_ind = -1
        for i in range(len(kernel_list)):
            if rand_dim in kernel_list[i].active_dims:
                old_ind = i
                break

        num_parts = len(kernel_list)
        old_kernel = kernel_list[old_ind]
        if len(old_kernel.active_dims) == 1:
            rand_ind_target = random.randrange(0, len(kernel_list))
            while rand_ind_target == old_ind:
                rand_ind_target = random.randrange(0, len(kernel_list))
            target_kernel = kernel_list[rand_ind_target]
            new_active_dims = torch.cat((target_kernel.active_dims, torch.tensor([rand_dim])))
            new_kernel = gpytorch.kernels.RBFKernel(log_lengthscale_bounds=(-10, 10), active_dims=new_active_dims)

            kernel_list = [v for i, v in enumerate(kernel_list) if i not in [old_ind, rand_ind_target]]
            kernel_list.append(new_kernel)
            proposal_logprob = -log(self.num_dims) - log(num_parts - 1)
            reverse_logprob = proposal_logprob
        else:
            old_active_dims = old_kernel.active_dims
            new_active_dims = np.setdiff1d(old_active_dims, [rand_dim])
            kernel1 = gpytorch.kernels.RBFKernel(log_lengthscale_bounds=(-10, 10), active_dims=new_active_dims)

            rand_ind_target = random.randrange(0, len(kernel_list))

            # if equal, split the kernel, otherwise merge into another
            if old_ind == rand_ind_target:
                kernel2 = gpytorch.kernels.RBFKernel(log_lengthscale_bounds=(-10, 10), active_dims=[rand_dim])
                kernel_list = [v for i, v in enumerate(kernel_list) if i not in [old_ind]]
            else:
                target_kernel = kernel_list[rand_ind_target]
                new_active_dims2 = torch.cat((target_kernel.active_dims, torch.tensor([rand_dim])))
                kernel2 = gpytorch.kernels.RBFKernel(log_lengthscale_bounds=(-10, 10), active_dims=new_active_dims2)
                kernel_list = [v for i, v in enumerate(kernel_list) if i not in [old_ind, rand_ind_target]]
            kernel_list.append(kernel1)
            kernel_list.append(kernel2)
            proposal_logprob = -log(self.num_dims) - log(num_parts)
            reverse_logprob = proposal_logprob
        return kernel_list, proposal_logprob, reverse_logprob


class KShiftDimAdditiveStructureProposer:
    def __init__(self, K):
        self.sd_proposal = ShiftDimAdditiveStructureProposer()
        self.K = K

    def propose(self, kernel_list):
        plp = []
        rlp = []
        new_kl, pl, rl = self.sd_proposal.propose(kernel_list)
        plp.append(pl)
        rlp.append(rl)
        for i in range(self.K - 1):
            new_kl, pl, rl = self.sd_proposal.propose(new_kl)
            plp.append(pl)
            rlp.append(rl)
        return new_kl, sum(plp), sum(rlp)


class BagofModelsAdditiveStructureSelector:
    def __init__(self, X, y, num_dims):
        if X is not None and not isinstance(X, torch.FloatTensor):
            X = torch.Tensor(X).type(torch.FloatTensor)  # .cuda()
        if y is not None and not isinstance(y, torch.FloatTensor):
            y = torch.Tensor(y).type(torch.FloatTensor).squeeze(1)  # .cuda()

        self.X = X
        self.y = y

        self.models = []
        self.trace = []

        self.proposer = KShiftDimAdditiveStructureProposer(num_dims)

        # initialize the very first kernel
        mcmc_kern = gpytorch.kernels.AdditiveKernel(
            gpytorch.kernels.RBFKernel(log_lengthscale_bounds=(-10, 10), active_dims=range(num_dims))
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood(log_noise_bounds=(-5, 1))  # .cuda()
        mcmc_model = AdditiveStructureGPModel(self.X, self.y, likelihood, mcmc_kern)  # .cuda()

        mcmc_model.train()
        mcmc_model.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([{"params": mcmc_model.parameters()}], lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(mcmc_model.likelihood, mcmc_model)

        training_iter = 300
        for i in range(training_iter):
            optimizer.zero_grad()
            output = mcmc_model(self.X)
            loss = -mll(output, self.y)
            mcmc_model.log_likelihood = -loss
            loss.backward()
            optimizer.step()

        self.current_model = mcmc_model
        self.current_kernel = mcmc_kern

    def set_sample(self, trace_value, model, kernel):
        self.trace.append(trace_value)
        self.models.append(model)
        self.current_model = model
        self.current_kernel = kernel

    def get_models(self, num_models):
        for i in range(num_models):
            # propose a kernel
            proposed_sample, _, _ = self.proposer.propose(self.current_kernel.kernels)
            proposed_kernel = gpytorch.kernels.AdditiveKernel(*proposed_sample)

            # construct a GP model
            likelihood = gpytorch.likelihoods.GaussianLikelihood(log_noise_bounds=(-5, 1))  # .cuda()
            proposed_model = AdditiveStructureGPModel(self.X, self.y, likelihood, proposed_kernel)  # .cuda()

            # train the model
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

            trace_value = proposed_model.log_likelihood
            self.set_sample(trace_value, proposed_model, proposed_kernel)

        # find best model
        best_model_idx = torch.argmax(torch.tensor(self.trace)).item()  # np.argmax(self.trace)
        best_model = self.models[best_model_idx]
        return best_model
