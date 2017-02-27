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
pars['beta'] = 10. # inverse (physical) temperature
pars['gamma'] = 1. # damping of (physical) Langevin dynamics
pars['upsilon'] = 1.e-3  # algorithmic time step
pars['chi'] = 1. # algorithmic friction for BAOAB scheme
pars['T'] = 1.  # end (physical) time
# pars['T'] = 1.e6  # end (physical) time
pars['p_dist'] = 1 # how are the initial and final p's distributed: p_0, p_N ~ N(0, p_dist/beta)

pars['N'] = 100  # number of time steps
pars['sigma'] = 1  # sigma = 1 samples phase space, sigma = 0 gives steepest descent

pars['periodic'] = True

orig_pars = pars.copy()


# the Strang_functions are (update_linear_nonlocal, update_nonlinear_nonlocal, update_nonlinear_local)
Strang_order = scipy.array((0,), dtype=int) # only performs the linear, nonlocal part!
# Strang_order = scipy.array((0,1,2), dtype=int)
# Strang_order = scipy.concatenate((Strang_order[:-1], Strang_order[::-1])) # adds the inverse order to be symmetric (order 2)
pars['Strang_order'] = Strang_order

pars['constrained'] = scipy.array((pars['N']/4, 3*pars['N']/4)) # which points are constrained
pars['constraints'] = scipy.array((1, -1)) # what the constrained values are

pot = potential.Potential()


N_experiments = 1
int_keys = ('N',) # for parsing the parameters
float_keys = ('beta', 'gamma', 'upsilon', 'T') # for parsing the paramters


U = 10 # number of algorithmic times to store
# Dt = 1.e-2 # gap between algorithmic times stored
Dt = pars['upsilon']
# alg_times is a vector of ALGORITHMIC times at which we want to store our solution
# alg_times = scipy.insert(scipy.power(10, scipy.arange(-2, 1, 1)), 0, 0)
alg_times = scipy.arange(U+1)*Dt # an
# alg_times = scipy.arange(U+1)*pars['upsilon']

# df = pandas.concat((pandas.DataFrame(index=range(N_experiments), columns=pars.keys() + ['mean_var_R',]), pandas.DataFrame(index=range(N_experiments), columns=alg_times)), axis=1)
series = pandas.Series(index=pars.keys() + ['mean_var_R',])
df = pandas.DataFrame(index=range(N_experiments), columns=pars.keys() + ['mean_var_R',])

# get = scipy.load('q0.npz')
# N_skip = get['q0'].shape[0]/pars['N']
# assert N_skip*pars['N'] == get['q0'].shape[0]
# q0 = get['q0'][::N_skip]


algs = (algorithm.AlgorithmOUConstraintsHamiltonian(pot, pars), \
        algorithm.AlgorithmTaylor(pot, pars), \
        algorithm.AlgorithmTaylorBAOAB(pot, pars), \
        algorithm.Algorithm(pot, pars), \
        algorithm.AlgorithmPseudospectralSemiImplicit(pot, pars), \
        algorithm.AlgorithmOUConstraints(pot, pars), \
        algorithm.AlgorithmQuasiPseudospectralSemiImplicit(pot, pars), \
        algorithm.AlgorithmSemiImplicit(pot, pars),\
        algorithm.AlgorithmInversePseudospectralSemiImplicit(pot, pars))

fig, axs = plt.subplots(1,2)

# for experiment in range(N_experiments):
experiment = 0
for alg_i in [0,]:
    alg = algs[alg_i]
    print experiment
    # if experiment > 0:
    #     for key in int_keys:
    #         pars[key] = 2*int(orig_pars[key]/2*scipy.exp(scipy.random.normal()))
    #     for key in float_keys:
    #         pars[key] = orig_pars[key]*scipy.exp(2*scipy.random.normal())
    # print pars

    # alg = algorithm.AlgorithmTaylorBAOAB(pot, pars)
    # alg = algorithm.AlgorithmOUConstraints(pot, pars)

    tau = pars['T']/pars['N']
    ts = scipy.arange(pars['N'])*tau
    # q0 = scipy.ones((pars['N'],))*1
    q0 = scipy.cos((ts/pars['T']-.25)*2*scipy.pi)
    qs = alg.iterate(q0, alg_times)

    Rs = scipy.zeros((3, U+1, pars['N']-2))
    Rs[0] = (qs[:,2:] - 2*qs[:,1:-1] + qs[:,:-2])/tau**2
    Rs[1] = pot.dV(qs[:,1:-1])
    Rs[2] = pars['gamma'] * (qs[:,2:]-qs[:,:-2])/(2*tau)
    Rs/= scipy.sqrt(2*pars['gamma']/(pars['beta']*tau))
    # plt.plot(Rs.var(axis=1))
    # plt.plot(experiment, Rs.var(axis=1).mean()-1.25, '.')

    for key, val in pars.items():
        try:
            series.loc[key] = val
        except ValueError:
            series.loc[key] = None
    series.loc['mean_var_R'] = Rs.sum(axis=0)[U/2:,:].var()
    df.loc[experiment] = series
    print series.loc['mean_var_R']
    print Rs.var(axis=(1,2))
    print Rs.sum(axis=0).var()

    df.to_csv('df_int.csv')

    axs[0].plot(alg_times, Rs.sum(axis=0).var(axis=-1).T)

    qs_hat = scipy.ifft(qs)
    abs_qs = abs(qs_hat).mean(axis=0)
    qs_plot = scipy.array(abs_qs[:pars['N']/2+1])
    qs_plot[1:-1]+= abs_qs[-1:pars['N']/2:-1]
    # axs[1].loglog(qs_plot)
    # axs[1].loglog(abs_qs, '--')
    axs[1].plot(ts, qs.T)

    plt.show(block=False)

# plt.figure()
# plt.plot(alg_times, Rs.sum(axis=0).mean(axis=1))
# plt.plot(alg_times, Rs.sum(axis=0).std(axis=1))
#
# plt.figure()
# plt.hist(Rs.sum(axis=0).flatten(), bins=40)
#
plt.show()