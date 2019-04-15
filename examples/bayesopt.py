import torch
import gpytorch
from torch.optim import Optimizer
from LBFGS import FullBatchLBFGS
from sklearn.preprocessing import StandardScaler


def scale_to_bounds(x, lbs, ubs):
    x = x - lbs
    x = x / (ubs - lbs)
    return x


def unscale_to_bounds(x, lbs, ubs):
    x = x * (ubs - lbs)
    x = x + lbs
    return x


def _init_model_from_vector(model, vec):
    for param in model.parameters():
        numel = param.numel()
        param.data = vec[:numel].view_as(param)
        vec = vec[numel:]


def _vector_from_params(params):
    # get 1 x d vector
    return torch.cat(list(p.data.view(-1, 1) for p in params)).transpose(-2, -1)


class BayesOptDerivativeOptimizer(Optimizer):
    def __init__(self, params, optim_lbs, optim_ubs):
        num_params = _vector_from_params(list(params)).numel()

        class GPModelWithDerivatives(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMeanGrad()
                self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=num_params)
                self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

        self.likelihood_class = gpytorch.likelihoods.MultitaskGaussianLikelihood
        self.gp_class = GPModelWithDerivatives
        self.num_params = num_params

        # Seen hyperparameter values
        self.train_x = None  # n x num_params
        self.train_y = None  # Should always be n x (num_params + 1)

        first_param = next(iter(params))
        self.cand_set = torch.rand(500, self.num_params)

        self.optim_lbs = torch.tensor(optim_lbs)
        self.optim_ubs = torch.tensor(optim_ubs)

        super().__init__(params, {})

    def step(self, closure):
        # Step 1: Fit GP model to current training data
        likelihood = self.likelihood_class(num_tasks=self.num_params + 1)

        if self.train_x is not None:
            with gpytorch.settings.fast_computations(log_prob=False, solves=False):
                # Train on scaled data
                scaled_train_x = scale_to_bounds(self.train_x, self.optim_lbs, self.optim_ubs).detach()
                scaled_train_y = torch.tensor(
                    StandardScaler().fit_transform(self.train_y),
                ).to(scaled_train_x.dtype)
                self.gp_model = self.gp_class(scaled_train_x, scaled_train_y, likelihood)
                self.gp_model.train()
                likelihood.train()

                # Step 2: fit GP hyperparameters
                sub_optimizer = FullBatchLBFGS(self.gp_model.parameters(), lr=0.1)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.gp_model)

                def sub_closure():
                    sub_optimizer.zero_grad()
                    output = self.gp_model(scaled_train_x)
                    loss = -mll(output, scaled_train_y)
                    return loss

                for i in range(20):  # Max 15 iterations of LBFGS
                    loss = sub_closure()
                    loss.backward()
                    options = {'closure': sub_closure, 'current_loss': loss, 'max_ls': 10}
                    loss, _, _, _, _, _, _, fail = sub_optimizer.step(options)
                    print(f'Inner iteration {i}, loss = {loss}')

                    if fail:
                        break
        else:
            self.gp_model = self.gp_class(None, None, likelihood)

        # Step 3: Thompson sampling
        with gpytorch.settings.fast_computations(log_prob=False, solves=False):
            self.gp_model.eval()
            predictions = self.gp_model(self.cand_set)
            fsample = predictions.sample(torch.Size([1]))
            best_cand = fsample[:, :, 0].min(dim=-1).indices.item()
            unscaled_next_pt = self.cand_set[best_cand]
            next_pt = unscale_to_bounds(unscaled_next_pt, self.optim_lbs, self.optim_ubs)

        # Step 4: Evaluate function
        print(next_pt)
        loss = closure(next_pt).view(1, 1).cpu()  # Assume closure calls backward

        # Update training data
        grads = [p.grad for p in self.param_groups[0]['params']]
        params = self.param_groups[0]['params']  # 0 is hardcoded b/c we expect a single input for params

        param_vec = _vector_from_params(params).detach().cpu()  # 1 x d
        grad_vec = _vector_from_params(grads).cpu()  # 1 x d

        new_y = torch.cat([loss, grad_vec], dim=-1).detach()  # 1 x (d + 1)

        if self.train_x is None:
            self.train_x = param_vec
            self.train_y = new_y
        else:
            self.train_x = torch.cat([self.train_x, param_vec])
            self.train_y = torch.cat([self.train_y, new_y])

        return loss
