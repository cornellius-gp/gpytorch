from .mean import Mean


class ConstantMean(Mean):
    def forward(self, input, constant):
        return constant.expand(input.size())
