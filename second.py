import scipy
from matplotlib import pyplot as plt
import seaborn as sns
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

upsilon = 5.e-5 # algorithmic time step

gL = +1
gR = -1
gM = (gR+gL)/2

qL = gL
qR = gR # these are the trivial inversions of g...

t0 = 0.
q0 = scipy.linspace(qL, qR, N+1)
# q0 = scipy.cos(ts/T*scipy.pi)
# t0 = 100.
# get = scipy.load('qs_beta{}_tau{}_time{}.npz'.format(beta, tau, t0))
# q0 = get['qs']


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

def update(qs):
    ddq_like = (qs[2:] - 2*qs[1:-1] + qs[:-2])/tau
    gradV_like = dV(qs)*tau
    hessV_like = ddV(qs)*tau
    gradU_const = scipy.zeros((N+1,))
    gradU_lin = scipy.zeros((N+1,))
    # gradU = scipy.zeros_like(qs)
    #
    # gradU[2:]  += (ddq_like + gradV_like[:-2])/tau
    # gradU[1:-1]+= (ddq_like + gradV_like[1:-1])*(-2/tau + hessV_like[1:-1])
    # gradU[:-2] += (ddq_like + gradV_like[2:])/tau
    # gradU      *= beta/(4*tau)

    # gradU[0] += beta/(tau+p_dist)*((qs[1] -qs[0] )/tau + .5*gradV_like[0] )*(-1/tau + .5*hessV_like[0])
    # gradU[1] += beta/(tau+p_dist)*((qs[1] -qs[0] )/tau + .5*gradV_like[0] )*( 1/tau)


    gradU_const[2:]  += ((-2*qs[1:-1] +   qs[:-2] )/tau + gradV_like[:-2])/tau
    gradU_const[1:-1]+= ((   qs[2:]   +   qs[:-2] )/tau + gradV_like[1:-1])*(-2/tau + hessV_like[1:-1])
    gradU_const[:-2] += ((   qs[2:]   - 2*qs[1:-1])/tau + gradV_like[2:])/tau
    gradU_const      *= beta/(4*tau)

    gradU_lin[2:]  +=  1/tau**2
    gradU_lin[1:-1]+= -2/tau*(-2/tau + hessV_like[1:-1])
    gradU_lin[:-2] +=  1/tau**2
    gradU_lin      *= beta/(4*tau)

    gradU_const[0] += beta/(tau+p_dist)*( qs[1]/tau + .5*gradV_like[0] )*(-1/tau + .5*hessV_like[0])
    gradU_const[1] += beta/(tau+p_dist)*(-qs[0]/tau + .5*gradV_like[0] )*( 1/tau)
    gradU_lin[0]   += beta/(tau+p_dist)*(-1/tau + .5*gradV_like[0] )*(-1/tau + .5*hessV_like[0])
    gradU_lin[1]   += beta/(tau+p_dist)*( 1/tau + .5*gradV_like[0] )*( 1/tau)

    # try:
    #     assert (abs(gradU_const + gradU_lin*qs - gradU) < 1.e-10).all()
    # except AssertionError:
    #     print gradU_const + gradU_lin*qs
    #     print gradU
    #     print (gradU_const + gradU_lin*qs - gradU)/gradU
    #     print (abs(gradU_const + gradU_lin*qs - gradU))[abs(gradU_const + gradU_lin*qs - gradU) >= 1.e-10]
    #     raise AssertionError

    gamma =  gradU_lin
    mu    = -gradU_const/gamma
    ou1 = scipy.exp(-gamma*upsilon)
    ou2 = 1.-ou1
    ou3 = scipy.sqrt(2*(scipy.exp(2*gamma*upsilon)-1)/(2*gamma))
    qs[:] = qs*ou1 + mu*ou2 + ou1*ou3*scipy.random.normal(size=qs.shape)
    # qs += - (gradU_const + gradU_lin*qs) *upsilon# + scipy.sqrt(2*upsilon)*scipy.random.normal(size=(N+1,))
    qs[0]   = gL
    qs[1]   = gL
    qs[-2]  = gR
    qs[-1]  = gR
    qs[N/2] = gM

# alg_times = scipy.insert(scipy.power(10, scipy.arange(-2, 1, 1)), 0, 0)
alg_times = scipy.arange(0, 1.0+upsilon, .1)
# alg_times = scipy.arange(0, 200.1*upsilon, upsilon)

# extra_times = scipy.arange(0, 105*upsilon, 10*upsilon)
# alg_times = scipy.concatenate((alg_times, alg_times[-1]+extra_times))
alg_times += t0

qs_save = scipy.zeros(alg_times.shape + q0.shape)
qs = scipy.array(q0) # make sure to copy the values

n = 0
for s in scipy.arange(t0, alg_times[-1]+.5*upsilon, upsilon):
    if s >= alg_times[n]:
        qs_save[n,:] = qs
        n += 1
        plt.plot(ts, qs_save[:n,:].T, '.')
        plt.show()
        print s

    update(qs)


