from .settings import _feature_flag


class fast_pred_var(_feature_flag):
    """
    Fast predictive variances - with Lanczos
    """
    _n_probe_vectors = 1

    @classmethod
    def n_probe_vectors(cls):
        return cls._n_probe_vectors

    @classmethod
    def _set_n_probe_vectors(cls, value):
        cls._n_probe_vectors = value

    def __init__(self, state=True, n_probe_vectors=1):
        self.orig_value = self.__class__.n_probe_vectors()
        self.value = n_probe_vectors
        super(fast_pred_var, self).__init__(state)

    def __enter__(self):
        self.__class__._set_n_probe_vectors(self.value)
        super(fast_pred_var, self).__enter__()

    def __exit__(self, *args):
        self.__class__._set_n_probe_vectors(self.orig_value)
        return super(fast_pred_var, self).__exit__()


class fast_pred_samples(_feature_flag):
    """
    Fast predictive samples - with Lanczos
    """
    pass


__all__ = [
    fast_pred_var,
    fast_pred_samples,
]
