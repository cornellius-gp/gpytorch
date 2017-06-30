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

class CategoricalRandomVariable(RandomVariable):
    def __init__(self, mass_function):
        self.mass_function = mass_function
        self._cumulative_mass_function = self.mass_function.cumsum()

    def representation(self):
        return self.mass_function

    def log_probability(self, i):
        if i > len(self.mass_function):
            raise RuntimeError('Attempted to access a Categorical mass function with a category number larger than the total number of categories: %d'.format(i))

        return math.log(self.mass_function[i])

    def sample(self):
        p = random.random()
        cmf_lt = self._cumulative_mass_function.ge(p)
        for i,v in enumerate(cmf_lt):
            if v == 1:
                return i

    def num_categories(self):
        return len(self.mass_function)

class IndependentRandomVariables(RandomVariable):
    def __init__(self, random_variable_list):
        self.random_variable_list = random_variable_list

    def log_probability(self, x):
        return sum([rv.log_probability() for rv in self.random_variable_list])

    def sample(self):
        return [rv.sample() for rv in self.random_variable_list]

    def __len__(self):
        return len(self.random_variable_list)

    def __iter__(self):
        for random_variable in self.random_variable_list:
            yield random_variable

    def __getitem__(self,i):
        return self.random_variable_list[i]


class BatchRandomVariables(IndependentRandomVariables):
    def __init__(self, random_variable, count):
        self.random_variable_list = [random_variable]*count

    def sample(self):
        return torch.cat([rv.sample().unsqueeze(0) for rv in self.random_variable_list])

class SamplesRandomVariable(RandomVariable):
    def __init__(self, sample_list):
        self._sample_list = sample_list

    def sample(self):
        ix = random.randint(0,len(self._sample_list))
        return self._sample_list[ix]

class CategoricalRandomVariable(RandomVariable):
    def __init__(self, mass_function):
        self.mass_function = mass_function
        self._cumulative_mass_function = self.mass_function.cumsum()

    def representation(self):
        return self.mass_function

    def log_probability(self, i):
        if i > len(self.mass_function):
            raise RuntimeError('Attempted to access a Categorical mass function with a category number larger than the total number of categories: %d'.format(i))

        return math.log(self.mass_function[i])

    def sample(self):
        p = random.random()
        cmf_lt = self._cumulative_mass_function.ge(p)
        for i,v in enumerate(cmf_lt):
            if v == 1:
                return i

    def num_categories(self):
        return len(self.mass_function)


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
