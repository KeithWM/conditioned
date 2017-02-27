import scipy
import scipy.fftpack
from matplotlib import pyplot as plt
import seaborn as sns
# import potential_dw as potential
import potential_none as potential
import algorithm
import pandas

sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2)
plt.close('all')
# plt.ion()

pars = {}
pars['beta'] = 1.
pars['gamma'] = 1.
pars['upsilon'] = 1.e-5  # algorithmic time step
pars['T'] = 10.  # end (physical) time
# pars['T'] = 1.e6  # end (physical) time
pars['p_dist'] = 1 # how are the initial and final p's distributed: p_0, p_N ~ N(0, p_dist/beta)

pars['N'] = 1000  # number of time steps
pars['sigma'] = 1  # sigma = 1 samples phase space, sigma = 0 gives steepest descent

pars['periodic'] = True

orig_pars = pars.copy()

# Strang_functions = (update_linear_nonlocal, update_nonlinear_nonlocal, update_nonlinear_local)
# Strang_order = scipy.array((0,1,2), dtype=int)
Strang_order = scipy.array((0,), dtype=int)
# Strang_order = scipy.concatenate((Strang_order, Strang_order[::-1]))
pars['Strang_order'] = Strang_order

pot = potential.Potential()

alg = algorithm.Algorithm(pot, pars)

omega = 4
tau = pars['T']/pars['N']
ts = scipy.arange(pars['N'])*tau
# q0 = scipy.ones((pars['N'],))*0
q0 = scipy.cos(omega*ts/pars['T']*2*scipy.pi)
q0_d2 = -(2*omega*scipy.pi/pars['T'])**2*q0
q0_d4 = (2*omega*scipy.pi/pars['T'])**4*q0

q1 = q0.copy()
q2 = q0.copy()
q0_hat = scipy.zeros_like(q0)
q1_hat = scipy.zeros_like(q0)
q2_hat = scipy.zeros_like(q0)

alg.transform(q0, q0_hat)
q1_hat = q0_hat * alg.d2
alg.inverse_transform(q1, q1_hat)
q2_hat = q0_hat * alg.d4
alg.inverse_transform(q2, q2_hat)

plt.plot(q0_d2)
plt.plot(q1,'--')
plt.plot(q0_d4)
plt.plot(q2,'--')

plt.show()