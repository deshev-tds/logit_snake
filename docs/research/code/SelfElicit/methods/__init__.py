class AbstractMethod:
    def __init__(self, args):
        self.args = args

    def __call__(self, sample, model, **kwargs):
        raise NotImplementedError
