import scipy
import scipy.linalg
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import spsolve

sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2)
plt.close('all')

scipy.random.seed(21)

beta = 10.
p_dist = 1. # how are the initial and final p's distributed: p_0, p_N ~ N(0, p_dist/beta)
upsilon = 5.e-5 # algorithmic time step

N = 100 # number of time steps
assert N%2 == 0 # to have a neat midpoint

T = 10. # end time
tau = T/N # time step
ts = scipy.linspace(0, T, N+1)


gL = +1
gR = -1
gM = (gR+gL)/2

qL = gL
qR = gR # these are the trivial inversions of g...

# q0 = scipy.linspace(qL, qR, N+1)
q0 = scipy.cos(ts/T*scipy.pi)


def V(q):
    return ((q-1)**2 * (q+1)**2)/(1+q**2)

def dV(x): # = -V'(x)
    return x*(8/(1+x**2)**2 - 2)

def ddV(x):
    return 8/(1+x**2)**2*(1 - 4*x**2/(1+x**2)) - 2

def dddV(x): # not sure I need this, but I had it from before...
    return -96*x/(1+x**2)**3 + 192*x**3/(1+x**2)**4

def g(q):
    return q

def dg(q):
    return 1

IMatrix = scipy.sparse.spdiags(scipy.ones(N+1), [0], N+1, N+1)

diags = scipy.zeros((5,N+1)) # tridiagonal matrix for solving q evo
# first, ignoring boundaries
diags[-2,:] =  1
diags[-1,:] = -4
diags[0,:]  =  6
diags[1,:]  = -4
diags[2,:]  =  1
# then the boundaries
diags[0,0] = 1
diags[1,1] = -2
diags[2,2] = 1

diags[-1,0] = -2
diags[0,1]  = 5
diags[1,2]  = -4

diags[1,-1]  = -2
diags[0,-2]  = 5
diags[-1,-1] = -4

diags[0,-1]  = 1
diags[-1,-2] = -2
diags[-2,-3] = 1
diags*= beta/(2*tau**3)

diags[-1,0] -= beta/(tau**2*(tau+p_dist))
diags[0,0]  += beta/(tau**2*(tau+p_dist))
diags[0,1]  += beta/(tau**2*(tau+p_dist))
diags[1,1]  -= beta/(tau**2*(tau+p_dist))
AMatrix = scipy.sparse.spdiags(diags, [0,1,2,-2,-1], N+1, N+1).tocsr()

diags = scipy.zeros((3,N+1))
diags[-1,:-2] = 1
diags[0,1:-1]  = -2
diags[1,2:]  = 1
diags*= beta/(2*tau)
diags[0,0]  -= beta/(2*(tau+p_dist))
diags[1,1]  += beta/(2*(tau+p_dist))
BMatrix = scipy.sparse.spdiags(diags, [0,1,-1], N+1, N+1).tocsr()

qs = scipy.array(q0)

def update(qs, qs_exp):
# if qs.all():
    ddq_like = (qs[2:] - 2*qs[1:-1] + qs[:-2])/tau
    gradV_like = dV(qs)
    hessV_like = ddV(qs)

    explicit = scipy.zeros_like(qs)
    explicit[2:]   += gradV_like[:-2]
    explicit[1:-1] += gradV_like[1:-1]*(-2 + hessV_like[1:-1]*tau**2)
    explicit[:-2]  += gradV_like[2:]
    explicit       *= beta/(2*tau)
    explicit[0] += beta/(tau+p_dist)*.5*gradV_like[0] * (-1 + .5*hessV_like[0]*tau**2)
    explicit[1] += beta/(tau+p_dist)*.5*gradV_like[0]

    explicit    += scipy.sqrt(2*upsilon)*scipy.random.normal(size=(N+1,))

    ddV_sparse = scipy.sparse.spdiags(hessV_like, [0], N+1,N+1)

    qs_old = scipy.array(qs)
    lhs = IMatrix - (AMatrix + ddV_sparse*BMatrix)*upsilon
    rhs = explicit*upsilon + qs


    # to remove the singularity
    lhs[0,1:3] = 0
    lhs[0,0] = 1
    rhs[0] = qL
    lhs[-1,-3:-1] = 0
    lhs[-1,-1] = 1
    rhs[-1] = qR

    # print lhs.toarray()

    qs[:] = spsolve(lhs, rhs)

    qs[0]   = gL
    qs[-1]  = gR
    qs[N/2] = gM

    qs_exp[:] = lhs*qs_exp - qs_exp + rhs
    qs_exp[0]   = gL
    qs_exp[-1]  = gR
    qs_exp[N/2] = gM

# alg_times = scipy.insert(scipy.power(10, scipy.arange(-2, 0, 1)), 0, 0)
alg_times = scipy.arange(0, .20+upsilon, .01)
# alg_times = scipy.arange(0, 21.1*upsilon, upsilon)

# extra_times = scipy.arange(0, 105*upsilon, 10*upsilon)
# alg_times = scipy.concatenate((alg_times, alg_times[-1]+extra_times))

qs_save = scipy.zeros(alg_times.shape + q0.shape)
qs = scipy.array(q0) # make sure to copy the values
qs_exp = scipy.array(q0) # make sure to copy the values

n = 0
for s in scipy.arange(0, alg_times[-1]+.5*upsilon, upsilon):
    if s >= alg_times[n]:
        plt.plot(ts, qs, 'o')
        plt.plot(ts, qs_exp, '.')
        qs_save[n,:] = qs[:]
        n += 1
        print s

    update(qs, qs_exp)

plt.show()

