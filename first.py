import scipy
import scipy.fftpack
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2)
plt.close('all')

N = 1000

pi = scipy.pi # pi, 1/2 of the ratio between the circumference and radius of a circle
sigma = 1 # noise
T = 1 # end time
tau = T/float(N) # real time step
upsilon = 1.e-4 # algorithmic time step
ts = scipy.linspace(0, T, N+1)

f = scipy.zeros((N+1,))
f_hat = scipy.zeros((N,))

xL = -0#1. # left BC
xR = +0#1. # right BC

"""
with these BCs, the zeroth eigenfunction is the particular solution
xL*cos(t/T(2*n+1)*pi) + xR*cos((T-t)/T(2*n+1)*pi)
with lambda_0 = (pi/(2*sigma*T))**2
"""

eigenvalues = scipy.arange(N)**2*pi**2/(sigma*T)**2
eigenvalues[0] = (pi/(2*sigma*T))**2

x_particular = xL * scipy.cos(ts / (2 * T) * pi) + xR * scipy.cos((T - ts) / (2 * T) * pi)

qs = scipy.ones((N,)) # qs determine the cylindrical Brownian motion

"""
Now follow some convenient preliminary computations
"""
coeff1 = scipy.exp(-eigenvalues*upsilon)
coeff2 = (1-coeff1)/eigenvalues
coeff3 = scipy.sqrt(qs/(2*eigenvalues)*(1 - scipy.exp(-2*eigenvalues*upsilon)))*N # factor N to account for fftpack.dst

def transform(x, x_hat):
    x_hat[1:] = scipy.fftpack.dst(x[1:-1] - x_particular[1:-1], type=1)

def inverseTransform(x, x_hat):
    x[:] = x_particular
    x[1:-1]+= scipy.fftpack.idst(x_hat[1:], type=1)/(2*N) # seems to be wrong by a factor 2 :-S

def V(x):
    return (x-1)**2*(x+1)**2/(1+x**2)

def g(x): # = -V'(x)
    return x*(8/(1+x**2)**2 - 2)

def dg(x):
    # return 8/(1+x**2)**2 - 2 - 32*x**2/(1+x**2)**3
    return 8/(1+x**2)**2*(1 - 4*x**2/(1+x**2)) - 2

def ddg(x):
    return -96*x/(1+x**2)**3 + 192*x**3/(1+x**2)**4

def f_function(x): # the nonlinear function f, acting on 'physical space'
    # return .5*x
    return -1/sigma**2*g(x)*dg(x) - .5*ddg(x)

def fN(x, x_hat, f, f_hat):
    inverseTransform(x, x_hat)
    f = f_function(x)
    transform(f, f_hat)

def stepExpEuler(x, x_hat, f, f_hat):
    fN(x, x_hat, f, f_hat)

    x_hat = coeff1*x_hat + coeff2*f_hat + coeff3*scipy.random.normal(size=(N,))
    x_hat[0] = 0
    return x_hat

x0 = scipy.zeros((N+1,))
x0_hat = scipy.zeros((N,))

# x0 = scipy.linspace(2, 0, N+1)
x0 = scipy.linspace(xL, xR, N+1)
# # x0+= scipy.sin(ts/T*pi)
# x0[0]  = xL
# x0[-1] = xR
transform(x0, x0_hat)
# x0_hat = N/scipy.arange(N, dtype=float)
# x0_hat[0] = 0
inverseTransform(x0, x0_hat)

x = x0.copy()
x_hat = x0_hat.copy()

# alg_times = scipy.arange(0, 10.01, 1.)
alg_times = scipy.insert(scipy.power(10, scipy.arange(-3, 0, 1)), 0, 0)

n = 0
for s in scipy.arange(0, alg_times[-1]+.5*upsilon, upsilon):
    if s >= alg_times[n]:
        plt.plot(ts, x, '.')
        n += 1
        print s

    x_hat = stepExpEuler(x, x_hat, f, f_hat)
    inverseTransform(x, x_hat)



plt.show()