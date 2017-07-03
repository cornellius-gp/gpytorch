from .random_variable import RandomVariable

class ConstantRandomVariable(RandomVariable):
    def __init__(self, value):
        self._value = value

    def sample(self):
        return self._value

    def __setitem__(self, key, value):
        self._value[key] = value

    def __getitem__(self, key):
        return self._value[key]

