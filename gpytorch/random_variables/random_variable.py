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
        if type(self) != type(other):
            raise RuntimeError('Can only add random variables of the same type')
        return self.__class__(*(a + b for a, b in zip(self.representation(), other.representation())))

    def __div__(self, other):
        return self.__mul__(1. / other)

    def __mul__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            raise RuntimeError('Can only multiply by scalars')
        return self.__class__(*(a * other for a in self.representation()))
