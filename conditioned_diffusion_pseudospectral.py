import scipy
import scipy.fftpack
from matplotlib import pyplot as plt
import seaborn as sns
# import potential_dw as potential
import potential_none as potential
import matrices

beta = 2.
gamma = .01
p_dist = 1. # how are the initial and final p's distributed: p_0, p_N ~ N(0, p_dist/beta)

upsilon = 1.e-5 # algorithmic time step
sigma = 1 # sigma = 1 samples phase space, sigma = 0 gives steepest descent

N = 10000 # number of time steps
assert N%4 == 0 # to have neat quarterpoints

T = 2.*scipy.pi # end time
tau = T/N # time step
# ts = scipy.linspace(0, T, N+1)
ts = scipy.arange(0, N)*tau
assert ts.shape[0] == N

gL = +1.
gM = -1
# gR = gL # not required in periodic case

qL = gL # these are the trivial inversions of g...
qM = gM
# qR = gR # not required in periodic case!

t0 = 0.
q0 = scipy.zeros((N,))
# q0 = scipy.sin(2*ts/T*scipy.pi)
# q0 = scipy.insert(scipy.cumsum(scipy.random.normal(size=(N-1,)))*scipy.sqrt(beta*tau/gamma), 0, 0)
# q0 = q0 - q0[-1]*ts/T
# t0 = 100.
# get = scipy.load('qs_beta{}_tau{}_time{}.npz'.format(beta, tau, t0))
# get = scipy.load('bestq0_beta{}_tau{}.npz'.format(beta, tau))
# q0 = get['q0']
# Nskip = (q0.shape[0]-1)/N
# q0 = q0[::Nskip]


qs = scipy.array(q0) # make sure to copy the values
qs_hat = scipy.zeros((N,)) # slightly smaller as end points are fixed at 0
fs = scipy.zeros_like(q0) # an array to store a force
fs_hat = scipy.zeros((N,)) # slightly smaller as end points are fixed at 0


# Strang_functions = (update_linear_nonlocal, update_nonlinear_nonlocal, update_nonlinear_local)
Strang_order = scipy.array((0,1,2), dtype=int)
Strang_order = scipy.array((0,), dtype=int)
# Strang_order = scipy.concatenate((Strang_order, Strang_order[::-1]))
upsilon_linear_nonlocal    = 0.
upsilon_nonlinear_nonlocal = 0.
upsilon_nonlinear_local    = 0.
if not (Strang_order==0).sum() == 0:
    upsilon_linear_nonlocal    = upsilon/(Strang_order==0).sum()
if not (Strang_order==1).sum() == 0:
    upsilon_nonlinear_nonlocal = upsilon/(Strang_order==1).sum()
if not (Strang_order==2).sum() == 0:
    upsilon_nonlinear_local    = upsilon/(Strang_order==2).sum()
upsilons = (upsilon_linear_nonlocal, upsilon_nonlinear_nonlocal, upsilon_nonlinear_local)

pot = potential.Potential()

# for the DST: (!)
# d2 = -(scipy.arange(N-1)+1)**2*(scipy.pi/T)**2
# # d4 =  (scipy.arange(N-1)+1)**4*(scipy.pi/T)**4
# d4 = d2**2
# for the (R)FFT:
d2_complex = -(2*scipy.pi/T*scipy.arange(N/2+1))**2
d2 = scipy.zeros((N,))
print d2.shape
d2[0::2] = d2_complex[:-1]
d2[1::2] = d2_complex[1:]
d4 = d2**2
theta = (d4 - gamma**2*d2)
ou1 = scipy.exp(-upsilon_linear_nonlocal*theta )
# ou2 = scipy.sqrt( 1*gamma*N*(1-ou1**2)/(beta*tau) )/scipy.sqrt(theta)
ou2 = scipy.sqrt(gamma*N*(1-ou1**2)/(beta*theta*tau))
ou2[0] = scipy.sqrt( gamma*N/(beta*tau) )

def g(q):
    return q

def dg(q):
    return 1

def transform(qs, qs_hat):
    qs_hat[:] = scipy.fftpack.rfft(qs, n=N)

def inverse_transform(qs, qs_hat):
    qs[:] = scipy.fftpack.irfft(qs_hat, n=N)

def constrain(qs):
    # qs[N/4]   = gL
    # qs[3*N/4] = gM
    return

def update_linear_nonlocal(qs, qs_hat, fs, fs_hat):
    # plt.plot(ts, qs)
    transform(qs, qs_hat)
    qs_hat[:] = ou1*qs_hat + ou2*scipy.random.normal(size=(N,))
    inverse_transform(qs, qs_hat)
    # plt.plot(ts, qs)
    # plt.show()
    return

def update_nonlinear_nonlocal(qs, qs_hat, fs, fs_hat):
    hessV_like = pot.ddV(qs)

    transform(qs, qs_hat)
    fs_hat[:] = qs_hat*d2
    inverse_transform(qs, qs_hat)
    inverse_transform(fs, fs_hat)
    fs*= -2*hessV_like
    # plt.plot(ts, fs,':')
    # plt.show()
    qs+=fs*upsilon_nonlinear_nonlocal
    return

def update_nonlinear_local(qs, qs_hat, fs, fs_hat):
    dVddV = scipy.zeros_like(qs) # only the terms like V'(q)V''(q)
    dVddV[1:-1] -= .5*pot.dV(qs)[1:-1]*pot.ddV(qs)[1:-1]
    qs+= dVddV * upsilon_nonlinear_local
    return


fig = plt.figure()
axs = (fig.add_subplot(1,2,1), fig.add_subplot(1,2,2))

def diagPlots(s, qs):
    # axs[0].plot(ts, qs, '-')
    Rs = (qs[2:] - 2*qs[1:-1] + qs[:-2])/tau**2 + pot.dV(qs[1:-1]) + gamma * (qs[2:]-qs[:-2])/(2*tau)
    # print (qs[2:] - 2*qs[1:-1] + qs[:-2])/tau**2
    # print pot.dV(qs[1:-1])
    # print gamma * (qs[2:]-qs[:-2])/(2*tau)
    print scipy.mean(Rs) * (beta*tau)/gamma, scipy.var(Rs) * (beta*tau) / gamma
    # axs[1].plot(ts[1:-1], Rs)
    # axs[1].plot(s, scipy.var(Rs) * (beta*tau) / gamma, '.')
    mean = scipy.mean(Rs)
    std = scipy.std(Rs)
    # axs[1].plot(scipy.array((s,s)), scipy.array((mean-std, mean+std)) * scipy.sqrt((beta*tau) / gamma), '-')
    # axs[1].plot(s, mean * scipy.sqrt((beta*tau) / gamma), 'o')
    # plt.hist(Rs, bins=40)
    # plt.show()
    # ps = (qs[1:]-qs[:-1])/tau
    # ax.plot(.5*(ts[1:]+ts[:-1]), ps, '-')
    # ax.plot(ts[1:-1], (ps[1:] - ps[:-1]) + pot.dV(qs[1:-1])*tau + ps[:-1]*gamma*tau, '-')
    # ax.plot(ts[1:-1], scipy.sqrt(2*tau/beta)*scipy.ones((N-1,)), '-')
    # ax.set_ylim([-2., 2.])

Strang_functions = (update_linear_nonlocal, update_nonlinear_nonlocal, update_nonlinear_local)

# if True:
def update(qs, qs_hat, fs, fs_hat):
    constrain(qs)
    for function_i in Strang_order:
        Strang_functions[function_i](qs, qs_hat, fs, fs_hat)


U = 5
Dt = .001
# alg_times = scipy.insert(scipy.power(10, scipy.arange(-2, 1, 1)), 0, 0)
alg_times = scipy.arange(0, U*Dt+upsilon, Dt)
# alg_times = scipy.arange(0, 0.1*upsilon, upsilon)

# extra_times = scipy.arange(0, 105*upsilon, 10*upsilon)
# alg_times = scipy.concatenate((alg_times, alg_times[-1]+extra_times))
# alg_times += t0

qs_save = scipy.zeros(alg_times.shape + q0.shape)


u = 0
for s in scipy.arange(t0, alg_times[-1]+.5*upsilon, upsilon):
    if s >= alg_times[u]:
        qs_save[u,:] = qs
        u += 1
        # plt.plot(ts, qs_save[:u,:].T, '.')
        # plt.show()
        diagPlots(s, qs)

        print s
    update(qs, qs_hat, fs, fs_hat)
