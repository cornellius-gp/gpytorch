import torch


class RandomVariable(object):
    def representation(self):
        raise NotImplementedError


    def mean(self):
        raise NotImplementedError


    def var(self):
        raise NotImplementedError


    def covar(self):
        raise NotImplementedError


    def sample(self, n_samples=1):
        raise NotImplementedError


    def std(self):
        return self.var().sqrt()


    def confidence_region(self):
        std2 = self.std().mul_(2)
        mean = self.mean()
        return mean.sub(std2), mean.add(std2)


class GaussianRandomVariable(RandomVariable):
    def __init__(self, mean, var):
        self._mean = mean
        self._var = var

    def __repr__(self):
        return repr(self.representation())

    def __len__(self):
        return self._mean.__len__()


    def representation(self):
        return self._mean, self._var


    def mean(self):
        return self._mean


    def covar(self):
        return self._var


    def var(self):
        return self.covar().diag()
