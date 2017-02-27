import scipy
import scipy.fftpack
from matplotlib import pyplot as plt
import seaborn as sns
import potential_dw as potential
# import potential_none as potential
import matrices

sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2)
plt.close('all')

scipy.random.seed(21)

beta = 1.
gamma = 1.
p_dist = 1. # how are the initial and final p's distributed: p_0, p_N ~ N(0, p_dist/beta)

upsilon = 1.e-2 # algorithmic time step
sigma = 1 # sigma = 1 samples phase space, sigma = 0 gives steepest descent

M = 1000
N = 10 # number of time steps
assert N%2 == 0 # to have a neat midpoint

T = 1. # end time
tau = T/N # time step
ts = scipy.linspace(0, T, N)
t0 = 0.
q0 = scipy.zeros((M, N,))


qs = scipy.array(q0) # make sure to copy the values
Qs = scipy.array(q0)
qs_hat = scipy.zeros_like(q0)
Qs_hat = scipy.zeros_like(q0)

def transform(qs, qs_hat):
    qs_hat[:] = scipy.fftpack.rfft(qs)

def inverse_transform(qs, qs_hat):
    qs[:] = scipy.fftpack.irfft(qs_hat)

ou = scipy.sqrt(2*upsilon*gamma/(beta*tau))
OU = scipy.ones((N,))*scipy.sqrt(N/2)*ou
OU[0]*=scipy.sqrt(2)
OU[-1]*=scipy.sqrt(2)

qs_hat_mean = scipy.array(abs(qs_hat).mean(axis=0))
Qs_hat_mean = scipy.array(abs(Qs_hat).mean(axis=0))

U = 10000

for u, s in enumerate(scipy.arange(0, U)*upsilon):
    qs+= ou*scipy.random.normal(size=(M,N,))
    Qs_hat+= OU*scipy.random.normal(size=(M,N,))

    transform(qs ,qs_hat)

    qs_hat_mean+= abs(qs_hat).mean(axis=0)
    Qs_hat_mean+= abs(Qs_hat).mean(axis=0)

    print qs_hat_mean[-1]/Qs_hat_mean[-1]

    if u*10%U == 0:
        plt.plot(qs_hat_mean/(u+1))
        plt.plot(Qs_hat_mean/(u+1), '--')
        print s

plt.show()
