# Base class for the engine object.


class EngineBase:
    def __init__(self, *args, **kwargs):
        pass

    def train(self):
        raise NotImplementedError(
            "La méthode 'train' doit être implémentée dans la classe dérivée."
        )

    def eval(self):
        raise NotImplementedError(
            "La méthode 'eval' doit être implémentée dans la classe dérivée."
        )

    def run(self):
        raise NotImplementedError(
            "La méthode 'run' doit être implémentée dans la classe dérivée."
        )
