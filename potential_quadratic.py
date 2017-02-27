class Potential(object):
    def __init__(self):
        pass

    def V(self, x):
        return .5*x**2

    def dV(self, x):
        return x

    def dV_lin(self, x): # = -V'(x)/x
        return x*0 + 1

    def ddV(self, x):
        return x*0

    def dddV(self, x): # not sure I need this, but I had it from before...
        return x*0