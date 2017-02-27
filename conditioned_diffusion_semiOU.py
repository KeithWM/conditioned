import scipy
from matplotlib import pyplot as plt
import seaborn as sns
import potential_dw as potential
import matrices

sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2)
plt.close('all')

scipy.random.seed(21)

beta = 10.
p_dist = 1. # how are the initial and final p's distributed: p_0, p_N ~ N(0, p_dist/beta)

N = 100 # number of time steps
assert N%2 == 0 # to have a neat midpoint
# assert not N%4 == 0 # to avoind odd-even decoupling???

T = 10. # end time
tau = T/N # time step
ts = scipy.linspace(0, T, N+1)

upsilon = 5.e-4 # algorithmic time step

gL = +1
gR = -1
gM = (gR+gL)/2

qL = gL
qR = gR # these are the trivial inversions of g...

t0 = 0.
# q0 = scipy.linspace(qL, qR, N+1)
q0 = scipy.cos(ts/T*scipy.pi)
# t0 = 100.
# get = scipy.load('qs_beta{}_tau{}_time{}.npz'.format(beta, tau, t0))
# q0 = get['qs']

qs = scipy.array(q0) # make sure to copy the values

pot = potential.Potential()
mat = matrices.Matrices(tau, beta, p_dist, N, upsilon, compute_expmA=True)

def g(q):
    return q

def dg(q):
    return 1


def update_linear(qs):
    qs[:] = mat.expmA*qs

# if True:
def update(qs):
    gradV_like = pot.dV(qs)
    hessV_like = pot.ddV(qs)

    qs_old = scipy.array(qs)
    update_linear(qs)
    qs_star = scipy.array(qs)

    explicit = scipy.zeros_like(qs)
    explicit[2:]   += gradV_like[:-2]
    explicit[1:-1] += gradV_like[1:-1]*(-2 + hessV_like[1:-1]*tau**2)
    explicit[:-2]  += gradV_like[2:]
    explicit       *= -beta/(2*tau)
    explicit[0] -= beta/(tau+p_dist)*.5*gradV_like[0] * (-1 + .5*hessV_like[0]*tau**2)
    explicit[1] -= beta/(tau+p_dist)*.5*gradV_like[0]

    # explicit    += scipy.sqrt(2/upsilon)*scipy.random.normal(size=(N+1,))

    ddV_sparse = scipy.sparse.spdiags(hessV_like, [0], N+1,N+1)

    lhs = mat.I + ddV_sparse*mat.B*upsilon
    rhs = explicit*upsilon + qs_star


    # to remove the singularity
    lhs[0,1:3] = 0
    lhs[0,0] = 1
    rhs[0] = qL
    lhs[-1,-3:-1] = 0
    lhs[-1,-1] = 1
    rhs[-1] = qR

    ddV_sparse = scipy.sparse.spdiags(hessV_like, [0], N+1,N+1)

    qs[:] = scipy.sparse.linalg.spsolve(lhs, rhs)
    qs[0]   = gL
    qs[-1]  = gR
    qs[N/2] = gM

# alg_times = scipy.insert(scipy.power(10, scipy.arange(-2, 1, 1)), 0, 0)
alg_times = scipy.arange(0, 1.0+upsilon, .1)
# alg_times = scipy.arange(0, 3.1*upsilon, upsilon)

# extra_times = scipy.arange(0, 105*upsilon, 10*upsilon)
# alg_times = scipy.concatenate((alg_times, alg_times[-1]+extra_times))
alg_times += t0

qs_save = scipy.zeros(alg_times.shape + q0.shape)

n = 0
for s in scipy.arange(t0, alg_times[-1]+.5*upsilon, upsilon):
    if s >= alg_times[n]:
        qs_save[n,:] = qs
        n += 1
        plt.plot(ts, qs_save[:n,:].T, '.')
        # plt.show()
        print s
    update(qs)
plt.show()


