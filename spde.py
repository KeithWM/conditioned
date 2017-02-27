import scipy

class Spde(object):
    def __init__(self):
        raise NotImplementedError

class ParabolicNonlinearSpde(Spde):
    def __init__(self):
        self.unboundedOperator = None
        self.nonlinearFunction = None
        self.noiseMatrix = None

    def initUnboundedOperator(self, ):