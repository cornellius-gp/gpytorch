class AcquisitionFunctionStrategy:
    def __init__(self, acquisition_function):
        self.acquisition_function = acquisition_function

    def maximize(self):
        raise NotImplementedError
