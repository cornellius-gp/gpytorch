class _feature_flag(object):
    _state = False

    @classmethod
    def on(cls):
        return cls._state

    @classmethod
    def _set_state(cls, state):
        cls._state = state

    def __init__(self):
        self.prev = self.__class__.on()

    def __enter__(self):
        self.__class__._set_state(True)

    def __exit__(self, *args):
        self.__class__._set_state(False)
        return False


class fast_pred_var(_feature_flag):
    """
    Fast predictive variances - with Lanczos
    """
    pass


class lanczos_preconditioners(_feature_flag):
    """
    Precondition CG with lanczos
    """
    pass
