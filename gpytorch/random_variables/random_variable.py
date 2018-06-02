from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Variable
from ..lazy import LazyVariable


class RandomVariable(object):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def confidence_region(self):
        """
        Returns 2 standard deviations above and below the mean.

        Return: tuple of two Variables (b x d) or (d), where b is the
        batch size and d is the dimensionality of the random variable

        First Variable is the lower end of the confidence region, second
        variable is the upper end
        """
        std2 = self.std().mul_(2)
        mean = self.mean()
        return mean.sub(std2), mean.add(std2)

    def cpu(self):
        new_args = []
        new_kwargs = {}
        for arg in self._args:
            if hasattr(arg, "cpu"):
                new_args.append(arg.cpu())
            else:
                new_args.append(arg)
        for name, val in self._kwargs.items():
            if hasattr(val, "cpu"):
                new_kwargs[name] = val.cpu()
            else:
                new_kwargs[name] = val
        return self.__class__(*new_args, **new_kwargs)

    def cuda(self, device_id=None):
        new_args = []
        new_kwargs = {}
        for arg in self._args:
            if hasattr(arg, "cuda"):
                new_args.append(arg.cuda(device_id))
            else:
                new_args.append(arg)
        for name, val in self._kwargs.items():
            if hasattr(val, "cuda"):
                new_kwargs[name] = val.cuda(device_id)
            else:
                new_kwargs[name] = val
        return self.__class__(*new_args, **new_kwargs)

    def covar(self):
        """
        Returns the covariance of the random variable

        Return: Variable (b x d x d) or (d x d), where b is the
        batch size and d is the dimensionality of the random variable
        """
        raise NotImplementedError

    def mean(self):
        """
        Returns the mean of the random variable

        Return: Variable (b x d) or (d), where b is the
        batch size and d is the dimensionality of the random variable
        """
        raise NotImplementedError

    def representation(self):
        """
        Returns a Variable (or tuple of Variables) that represent sufficient
        statistics of the Random variable
        """
        raise NotImplementedError

    def sample(self, n_samples=1):
        """
        Draw samples from the random variable.

        n_samples - number of samples to draw (default 1)

        Returns: tensor of samples

        If the random variable is in batch mode, result will be
        s x b x d, where s is the number of samples, and b is the batch size
        and d is the dimensionality of the random variable

        If the random variable is not in batch mode, result will be
        s x ..., where s is the number of samples
        and d is the dimensionality of the random variable
        """
        raise NotImplementedError

    def std(self):
        """
        Returns the standard deviation of the random variable

        Return: Variable (b x d) or (d), where b is the
        batch size and d is the dimensionality of the random variable
        """
        return self.var().sqrt()

    def var(self):
        """
        Returns the variance of the random variable

        Return: Variable (b x d) or (d), where b is the
        batch size and d is the dimensionality of the random variable
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns the batch size of the lazy variable
        """
        raise NotImplementedError

    def __repr__(self):
        return repr(self.representation())

    def __add__(self, other):
        raise NotImplementedError

    def __div__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __setattr__(self, name, val):
        if torch.is_tensor(val) or isinstance(val, Variable) or isinstance(val, LazyVariable):
            if not hasattr(self, "_args"):
                raise RuntimeError(
                    "Cannot assign {name} to LazyVariable before calling " "LazyVariable.__init__()".format(name=name)
                )
        object.__setattr__(self, name, val)
