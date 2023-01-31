#!/usr/bin/env python3

from ._blocker import BaseBlocker
from .k_means_blocker import KMeansBlocker
from .voronoi_blocker import VoronoiBlocker

from .ordering_strategies import OrderingStrategies

__all__ = [
    "BaseBlocker",
    "KMeansBlocker",
    "VoronoiBlocker",
    "OrderingStrategies"
]


# TODO: Where to put this???
def compute_mean_covar(blocks, x1, x2, y, mean_module, covar_module, training):

    # extract relevant info from blocks
    n_blocks = len(blocks.blocks)
    n_neighbors = len(blocks.neighbors[-1])

    # create empty lists to hold block means and covariances
    mean_list = []
    cov_list = []

    if training:
        # append mean function applied to first block in first spot
        mean_list.append(mean_module(x1[blocks.blocks[0]]))
        # append within covariance block to first spot
        cov_list.append(covar_module(x1[blocks.blocks[0]], x2[blocks.blocks[0]]))

        if n_neighbors == 0:
            # if no neighbors, all blocks are independent, so simply evaluate mean and covariance for each block
            for i in range(1, n_blocks):
                mean_list.append(mean_module(x1[blocks.blocks[i]]))
                cov_list.append(covar_module(x1[blocks.blocks[i]], x2[blocks.blocks[i]]))

        else:
            for i in range(1, n_blocks):
                # these calculations come from bottom of P7, Quiroz et al, 2021
                c_within = covar_module(x1[blocks.blocks[i]], x2[blocks.blocks[i]])
                c_between = covar_module(x1[blocks.blocks[i]], x2[blocks.neighbors[i]])
                c_neighbors = covar_module(x1[blocks.neighbors[i]], x2[blocks.neighbors[i]])

                # use cholesky decomposition to compute inverse, may be numerically unstable with large n_neighbors
                l_inv = c_neighbors.cholesky().inverse()
                # compute mean
                b = c_between @ l_inv.t() @ l_inv
                mean = mean_module(x1[blocks.blocks[i]]) + \
                       b @ (y[blocks.neighbors[i]] - mean_module(x2[blocks.neighbors[i]]))
                # compute covariance
                f = c_within - (c_between @ l_inv.t() @ l_inv @ c_between.t())

                mean_list.append(mean)
                cov_list.append(f)

    else:
        for i in range(0, len(blocks.blocks)):
            c_within = covar_module(x1[blocks.test_blocks[i]], x1[blocks.test_blocks[i]])
            c_between = covar_module(x1[blocks.test_blocks[i]], x2[blocks.test_neighbors[i]])
            c_neighbors = covar_module(x2[blocks.test_neighbors[i]], x2[blocks.test_neighbors[i]])

            # use cholesky decomposition to compute needed terms, may be numerically unstable with large n_neighbors
            l_inv = c_neighbors.cholesky().inverse()
            # compute mean
            b = c_between @ l_inv.t() @ l_inv
            mean = mean_module(x1[blocks.test_blocks[i]]) + \
                   b @ (y[blocks.test_neighbors[i]] - mean_module(x2[blocks.test_neighbors[i]]))
            # compute covariance
            f = c_within - (c_between @ l_inv.t() @ l_inv @ c_between.t())

            mean_list.append(mean)
            cov_list.append(f)

    return mean_list, cov_list
