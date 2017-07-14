class RandomVariable(object):
    def representation(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def var(self):
        raise NotImplementedError

    def covar(self):
        raise NotImplementedError

    def log_probability(self, x):
        raise NotImplementedError

    def sample(self, n_samples=1):
        raise NotImplementedError

    def std(self):
        return self.var().sqrt()

    def confidence_region(self):
        std2 = self.std().mul_(2)
        mean = self.mean()
        return mean.sub(std2), mean.add(std2)
