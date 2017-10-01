class RandomVariable(object):
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
        representation = self.representation()
        if not isinstance(representation, tuple) or isinstance(representation, list):
            representation = representation,
        return self.__class__(*(var.cpu() for var in representation))

    def cuda(self, device_id=None):
        representation = self.representation()
        if not isinstance(representation, tuple) or isinstance(representation, list):
            representation = representation,
        return self.__class__(*(var.cuda(device_id) for var in representation))

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
