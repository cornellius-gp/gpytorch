class _feature_flag(object):
    _state = False

    @classmethod
    def on(cls):
        return cls._state

    @classmethod
    def _set_state(cls, state):
        cls._state = state

    def __init__(self, state=True):
        self.prev = self.__class__.on()
        self.state = state

    def __enter__(self):
        self.__class__._set_state(self.state)

    def __exit__(self, *args):
        self.__class__._set_state(self.prev)
        return False


class fast_pred_var(_feature_flag):
    """
    Fast predictive variances - with Lanczos
    """
    pass


__all__ = [
    fast_pred_var,
]
