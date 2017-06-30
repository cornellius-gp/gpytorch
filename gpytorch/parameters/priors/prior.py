class Prior(object):
    def forward(self, x):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)