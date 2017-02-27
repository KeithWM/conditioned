import scipy
import scipy.fftpack
from matplotlib import pyplot as plt
import seaborn as sns
import potential_dw as potential
import matrices

sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2) 
plt.close('all')

scipy.random.seed(21)

beta = 5.
gamma = 1.
p_dist = 1. # how are the initial and final p's distributed: p_0, p_N ~ N(0, p_dist/beta)

M = 100 # number of ensemble members
N = 10000 # number of time steps
assert N%2 == 0 # to have a neat midpoint
# assert not N%4 == 0 # to avoind odd-even decoupling???

T = 100. # end time
tau = T/N # time step
ts = scipy.linspace(0, T, N+1)
ou1 = scipy.exp(-gamma*tau)
ou2 = scipy.sqrt((1-scipy.exp(-2*gamma*tau))/beta)

gL = +1
qL = gL

t0 = 0.
q0 = scipy.ones((M,))*qL
p0 = p_dist/beta*scipy.random.normal(size=(M,))
q0[:] = p0

q_space = scipy.linspace(-4,4,100)

qs = scipy.zeros((N+1,M))
ps = scipy.zeros((N+1,M))

q = q0.copy() # make sure to copy the values
p = p0.copy() # make sure to copy the values

pot = potential.Potential()

def g(q):
    return q

def dg(q):
    return 1

for n in range(N+1):
    print n, ' ',
    qs[n,:] = q
    ps[n,:] = p

    q+= p*.5*tau
    p+= -pot.dV(q)*.5*tau
    p[:] = p*ou1 + ou2*scipy.random.normal(size=(M,))
    p+= -pot.dV(q)*.5*tau
    q+= p*.5*tau

print ''
# plt.plot(ts, qs[:,:8])
# plt.show()

Rs = (qs[2:,:] - 2*qs[1:-1,:] + qs[:-2,:])/tau**2 + pot.dV(qs[1:-1,:]) + gamma * (qs[2:,:]-qs[:-2,:])/(2*tau)
print scipy.mean(Rs, axis=0)
print scipy.var(Rs, axis=0)
print gamma/(beta*tau)