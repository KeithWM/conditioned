class Potential(object):
    def __init__(self):
        pass

    def V(self, q):
        return ((q-1)**2 * (q+1)**2)/(1+q**2)

    def dV(self, x):
        return -x*(8/(1+x**2)**2 - 2)

    def dV_lin(self, x): # = -V'(x)/x
        return 8/(1+x**2)**2 - 2

    def ddV(self, x):
        return -8/(1+x**2)**2*(1 - 4*x**2/(1+x**2)) + 2

    def dddV(self, x): # not sure I need this, but I had it from before...
        return -96*x/(1+x**2)**3 + 192*x**3/(1+x**2)**4