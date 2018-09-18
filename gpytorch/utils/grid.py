from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def scale_to_bounds(x, lower_bound, upper_bound):
    """
    Scale the input data so that it lies in between the lower and upper bounds.

    Args:
        :attr:`x` (Tensor `n` or `b x n`):
            the input
        :attr:`lower_bound` (float)
        :attr:`upper_bound` (float)

    Returns:
        :obj:`torch.Tensor`
    """
    # Scale features so they fit inside grid bounds
    min_val = x.min()
    max_val = x.max()
    diff = max_val - min_val
    x = (x - min_val) * (0.95 * (upper_bound - lower_bound) / diff) + 0.95 * lower_bound
    return x


def choose_grid_size(train_inputs, ratio=1.):
    """
    Given some training inputs, determine a good grid size for KISS-GP.

    Args:
        :attr:`train_inputs` (Tensor `n` or `n x d` or `b x n x d`):
            training data
        :attr:`ratio` (float, optional):
            Ratio - number of grid points to the amount of data (default: 1.)

    Returns:
        :obj:`int`
    """
    # Scale features so they fit inside grid bounds
    num_data = train_inputs.numel() if train_inputs.dim() == 1 else train_inputs.size(-2)
    return int(ratio * num_data)
