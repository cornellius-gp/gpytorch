from torch import FloatTensor

class DualPrecisionTensor(FloatTensor):
    def __init__(self, float_tensor):
        self.float_tensor = float_tensor.float()
        self.half_tensor = float_tensor.half()

    def half(self):
        return self.half_tensor

    def __getattr__(self, name):
        return self.float_tensor.__getattr__(name)
