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

beta = 10.
gamma = 1.
p_dist = 1. # how are the initial and final p's distributed: p_0, p_N ~ N(0, p_dist/beta)

upsilon = 1.e-4 # algorithmic time step
sigma = 1 # sigma = 1 samples phase space, sigma = 0 gives steepest descent

N = 1000 # number of time steps
assert N%2 == 0 # to have a neat midpoint

T = 10. # end time
tau = T/N # time step
ts = scipy.linspace(0, T, N+1)

gL = -1.
gR = -gL
gM = (gR+gL)/2

qL = gL
qR = gR # these are the trivial inversions of g...

t0 = 0.
q0 = scipy.linspace(qL, qR, N+1)
# q0 = scipy.cos(ts/T*scipy.pi)
# t0 = 100.
# get = scipy.load('qs_beta{}_tau{}_time{}.npz'.format(beta, tau, t0))
# get = scipy.load('bestq0_beta{}_tau{}.npz'.format(beta, tau))
# q0 = get['q0']
# Nskip = (q0.shape[0]-1)/N
# q0 = q0[::Nskip]


qs = scipy.array(q0) # make sure to copy the values
qs_hat = scipy.zeros((N-1,)) # slightly smaller as end points are fixed at 0
fs = scipy.zeros_like(q0) # an array to store a force
fs_hat = scipy.zeros((N-1,)) # slightly smaller as end points are fixed at 0


# Strang_functions = (update_linear_nonlocal, update_nonlinear_nonlocal, update_nonlinear_local)
# Strang_order = scipy.arange(3, dtype=int)
Strang_order = scipy.array((0,1,1,2), dtype=int)
Strang_order = scipy.concatenate((Strang_order, Strang_order[::-1]))
upsilon_linear_nonlocal    = upsilon/(Strang_order==0).sum()
upsilon_nonlinear_nonlocal = upsilon/(Strang_order==1).sum()
upsilon_nonlinear_local    = upsilon/(Strang_order==2).sum()
upsilons = (upsilon_linear_nonlocal, upsilon_nonlinear_nonlocal, upsilon_nonlinear_local)

pot = potential.Potential()

# q_particular = (qL + (qR-qL)/T*ts)
# assert abs(qL + qR) < 1.e-14
q_particular = qL*scipy.cos(ts/T*scipy.pi)
d2 = -(scipy.arange(N-1)+1)**2*(scipy.pi/T)**2
# d4 =  (scipy.arange(N-1)+1)**4*(scipy.pi/T)**4
d4 = d2**2
ou1 = scipy.exp(upsilon_linear_nonlocal* (- d4 + gamma**2*d2) )
ou2 = scipy.sqrt( 2*gamma*N*(1-ou1**2)/(beta*tau) )/(-d2)

def g(q):
    return q

def dg(q):
    return 1

def transform(qs, qs_hat):
    qs_hat[:] = scipy.fftpack.dst(qs[1:-1] - q_particular[1:-1], type=1)

def inverse_transform(qs, qs_hat):
    qs[:]     = q_particular
    qs[1:-1] += scipy.fftpack.idst(qs_hat, type=1)/(2*N)

def constrain(qs):
    # qs[0]   = gL
    # qs[-1]  = gR
    # qs[N/2] = gM
    return

def update_linear_nonlocal(qs, qs_hat, fs, fs_hat):
    # transform(qs, qs_hat)
    # qs_hat[:] = ou1*qs_hat + ou2*scipy.random.normal(size=(N-1,))
    # inverse_transform(qs, qs_hat)
    return

def update_nonlinear_nonlocal(qs, qs_hat, fs, fs_hat):
    hessV_like = pot.ddV(qs)

    transform(qs, qs_hat)
    fs_hat[:] = qs_hat*d2
    inverse_transform(qs, qs_hat)
    inverse_transform(fs, fs_hat)
    plt.plot(qs, fs*upsilon_nonlinear_nonlocal, '-')
    fs*= -2*hessV_like
    plt.plot(qs, -2*hessV_like*upsilon_nonlinear_nonlocal)
    plt.plot(qs, fs*upsilon_nonlinear_nonlocal, '-')
    print -2*hessV_like
    plt.show()
    qs+=fs*upsilon_nonlinear_nonlocal
    constrain(qs)
    return

def update_nonlinear_local(qs, qs_hat, fs, fs_hat):
    # gradV_like = pot.dV(qs)
    # hessV_like = pot.ddV(qs)
    #
    # dVddV = scipy.zeros_like(qs) # only the terms like V'(q)V''(q)
    # dVddV[1:-1] -= .5*gradV_like[1:-1]*hessV_like[1:-1]
    # qs+= dVddV * upsilon_nonlinear_local
    # constrain(qs)
    return

Strang_functions = (update_linear_nonlocal, update_nonlinear_nonlocal, update_nonlinear_local)

# if True:
def update(qs, qs_hat, fs, fs_hat):
    for function_i in Strang_order:
        Strang_functions[function_i](qs, qs_hat, fs, fs_hat)


U = 100
Dt = .01
# alg_times = scipy.insert(scipy.power(10, scipy.arange(-2, 1, 1)), 0, 0)
# alg_times = scipy.arange(0, U*Dt+upsilon, Dt)
alg_times = scipy.arange(0, 0.1*upsilon, upsilon)

# extra_times = scipy.arange(0, 105*upsilon, 10*upsilon)
# alg_times = scipy.concatenate((alg_times, alg_times[-1]+extra_times))
alg_times += t0

qs_save = scipy.zeros(alg_times.shape + q0.shape)


fig = plt.figure()
axs = (fig.add_subplot(1,2,1), fig.add_subplot(1,2,2))

def diagPlots(s, qs):
    axs[0].plot(ts, qs, '-')
    Rs = (qs[2:] - 2*qs[1:-1] + qs[:-2])/tau**2 + pot.dV(qs[1:-1]) + gamma * (qs[2:]-qs[:-2])/(2*tau)
    print scipy.mean(Rs) * (beta*tau)/gamma, scipy.var(Rs) * (beta*tau) / gamma
    # axs[1].plot(ts[1:-1], Rs)
    # axs[1].plot(s, scipy.var(Rs) * (beta*tau) / gamma, '.')
    mean = scipy.mean(Rs)
    std = scipy.std(Rs)
    axs[1].plot(scipy.array((s,s)), scipy.array((mean-std, mean+std)) * (beta*tau) / gamma, '-')
    axs[1].plot(s, mean * (beta*tau) / gamma, 'o')
    # ps = (qs[1:]-qs[:-1])/tau
    # ax.plot(.5*(ts[1:]+ts[:-1]), ps, '-')
    # ax.plot(ts[1:-1], (ps[1:] - ps[:-1]) + pot.dV(qs[1:-1])*tau + ps[:-1]*gamma*tau, '-')
    # ax.plot(ts[1:-1], scipy.sqrt(2*tau/beta)*scipy.ones((N-1,)), '-')
    # ax.set_ylim([-2., 2.])


u = 0
for s in scipy.arange(t0, alg_times[-1]+.5*upsilon, upsilon):
    if s >= alg_times[u]:
        qs_save[u,:] = qs
        u += 1
        # plt.plot(ts, qs_save[:u,:].T, '.')
        # plt.show()
        # diagPlots(s, qs)
        print s
    update(qs, qs_hat, fs, fs_hat)

# print upsilon
# print qs_save.var(axis=0).mean()
# print (qs-q0)[:5]
# ax.plot(ts, qs_save[:u,:].T, '-')

# ax.plot(ts, qs)
# update_nonlinear(qs)
# ax.plot(ts, qs)
# update_linear_AB(qs, qs_hat, fs, fs_hat)
# ax.plot(ts, qs)

plt.show()