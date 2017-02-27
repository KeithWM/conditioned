import scipy
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2)
plt.close('all')
# plt.ion()

ls = scipy.arange(32, dtype=int)
facs = scipy.zeros((4, ls.shape[0]), dtype=float)
facs2 = scipy.zeros((4, ls.shape[0]), dtype=float)
tau = .1

fig, axs = plt.subplots(2,2)

N = 1024
# T = 32.
# factorial = scipy.zeros_like(ls, dtype=scipy.uint64)
factorial = scipy.zeros_like(ls, dtype=float)
factorial2 = scipy.zeros_like(ls, dtype=float)
for i_l, l in enumerate(ls):
    try:
        factorial[i_l] = scipy.misc.factorial(2*l + 4, exact=True)
        factorial2[i_l] = scipy.misc.factorial(2*l + 2, exact=True)
    except OverflowError:
        # factorial[i_l] = -1 # to get largest int...
        factorial[i_l] = scipy.misc.factorial(2*l + 4, exact=False)
        factorial2[i_l] = scipy.misc.factorial(2*l + 2, exact=False)

facs[1,:] = ( 2**(2*ls+5) - 8 )/( factorial )
facs2[1,:] = ( 2**(2*ls+1) )/( factorial2 )

for k in 2**scipy.arange(0, scipy.log2(N*2), 1, dtype=int):
    print k
    facs[0,:] = (-1)**ls
    facs[2,:] = (2 * scipy.pi * k / N) ** (2 * ls)
    facs[3,:] = scipy.prod(facs[:3,:], axis=0)

    facs2[0,:] = (-1)**ls
    facs2[2,:] = (2 * scipy.pi * k / N) ** (2 * ls)
    facs2[3,:] = scipy.prod(facs2[:3,:], axis=0)

    axs[0][0].set_prop_cycle(None)
    axs[0][0].semilogy(abs(facs.T), '-')
    axs[0][0].semilogy((-facs.T), 'k.')

    axs[1][0].set_prop_cycle(None)
    axs[1][0].semilogy(abs(facs2.T), '-')
    axs[1][0].semilogy((-facs2.T), 'k.')

    cum_facs = facs[3,:].cumsum()
    cum_facs2 = facs2[3,:].cumsum()

    # axs[1].set_prop_cycle(None)
    # axs[1].plot((cum_facs), '-')
    # axs[1].semilogy(abs(1-cum_facs), '-')
    # axs[1].semilogy((-cum_facs), 'k.')

    exact = (2 * scipy.sin(scipy.pi * k / N) * N / (2 * scipy.pi * k)) ** 4
    exact2 = (scipy.sin(2 * scipy.pi * k / N) * N / (2 * scipy.pi * k)) ** 2

    axs[0][1].loglog(k, 1 - exact, 'ro')
    axs[0][1].loglog(k, abs(1-cum_facs[-1]), 'k.')

    axs[1][1].loglog(k, 1 - exact2, 'ro')
    axs[1][1].loglog(k, abs(1-cum_facs2[-1]), 'k.')

ks = scipy.arange(1, 2 * N)
Ss = (2 * scipy.sin(scipy.pi * ks / N) * N / (2 * scipy.pi * ks)) ** 4
Ss2 = (scipy.sin(2 * scipy.pi * ks / N) * N / (2 * scipy.pi * ks)) ** 2

axs[0][1].loglog(ks, 1-Ss)

axs[1][1].loglog(ks, 1-Ss2)

plt.show()