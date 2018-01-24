class fast_pred_var(object):
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
