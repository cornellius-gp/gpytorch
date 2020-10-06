import torch
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from math import floor
import numpy as np
import pandas as pd


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_uci_data(data_dir, dataset, seed):
    set_seed(seed)

    data = torch.Tensor(loadmat(data_dir + dataset + '.mat')['data'])
    X = data[:, :-1]

    good_dimensions = X.var(dim=-2) > 1.0e-10
    if int(good_dimensions.sum()) < X.size(1):
        print("Removed %d dimensions with no variance" % (X.size(1) - int(good_dimensions.sum())))
        X = X[:, good_dimensions]

    if dataset in ['keggundirected', 'slice']:
        X = torch.Tensor(SimpleImputer(missing_values=np.nan).fit_transform(X.data.numpy()))

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y = data[:, -1]
    y -= y.mean()
    y /= y.std()

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.75 * X.size(0)))
    valid_n = int(floor(0.10 * X.size(0)))

    train_x = X[:train_n, :].contiguous().cuda()
    train_y = y[:train_n].contiguous().cuda()

    valid_x = X[train_n:train_n+valid_n, :].contiguous().cuda()
    valid_y = y[train_n:train_n+valid_n].contiguous().cuda()

    test_x = X[train_n+valid_n:, :].contiguous().cuda()
    test_y = y[train_n+valid_n:].contiguous().cuda()

    print("Loaded data with input dimension of {}".format(test_x.size(-1)))

    return train_x, train_y, test_x, test_y, valid_x, valid_y, None
