from torch import Tensor

class DualPrecisionTensor(Tensor):
    def __init__(self, float_tensor):
        self.float_tensor = float_tensor.float()
        self.half_tensor = float_tensor.half()

    def half(self):
        return self.half_tensor

    def __getattr__(self, name):
        return float_tensor.__getattr__(name)
