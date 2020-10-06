import os
import argparse
import uuid
from scipy.cluster.vq import kmeans2
import pickle
import time
import numpy as np
import math

import torch
from torch.distributions import Normal

import gpytorch

from load_uci_data import load_uci_data, set_seed
#from custom_loader import BatchDataloader


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_samples=80):
        print("WLSH kernel with num_samples = %d" % num_samples)
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.WLSHKernel(
           num_samples=num_samples, ard_num_dims=train_x.size(-1), num_dims=train_x.size(-1),
        ))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def main(args):
    train_x, train_y, test_x, test_y, valid_x, valid_y, y_std = \
        load_uci_data('./', args.d, args.seed)

    N_train = train_x.size(0)

    print("N_train", N_train)

    #train_loader = BatchDataloader(train_x, train_y, args.batch_size, shuffle=True)

    #inducing_points = (train_x[torch.randperm(N_train)[0:args.num_ind], :])
    #inducing_points = inducing_points.clone().data.cpu().numpy()
    #inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(),
    #                               inducing_points, minit='matrix')[0]).cuda()

    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = ExactGPModel(train_x, train_y, likelihood).cuda()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    likelihood.train()
    model.train()

    adam = torch.optim.Adam(model.parameters(), lr=0.02)
    sched = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=[300, 500], gamma=0.2)

    num_epochs = 600

    report_frequency = 20
    ts = [time.time()]

    for i in range(num_epochs):
        adam.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        adam.step()
        ts.append(time.time())

        if i % report_frequency == 0 or i == num_epochs - 1:
            model.eval()
            likelihood.eval()

            dt = 0.0 if i == 0 else (ts[-1] - ts[-2]) / float(report_frequency)

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                dist = likelihood(model(test_x))
                loc, covar = dist.loc, dist.covariance_matrix
                log_prob = Normal(loc, covar.diagonal().sqrt()).log_prob(test_y).mean().item()
                rmse = (dist.loc - test_y).pow(2.0).mean().sqrt().item()

            print("[Step %04d]  Loss: %.5f   Test RMSE: %.4f  Test LL: %.4f     [dt: %.3f]" % (i,
                  loss.item(), rmse, log_prob, dt))

            likelihood.train()
            model.train()

        sched.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-d', '--d', type=str, default='bike')
    parser.add_argument('-b', '--batch-size', type=int, default=100)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-ni', '--num-ind', type=int, default=100)
    args = parser.parse_args()

    main(args)
