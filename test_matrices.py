import matrices

pars = {}
pars['beta'] = 1.
pars['gamma'] = 1.
pars['upsilon'] = 1.e-4  # algorithmic time step
pars['T'] = 100.  # end (physical) time
pars['p_dist'] = 1 # how are the initial and final p's distributed: p_0, p_N ~ N(0, p_dist/beta)

pars['N'] = 100  # number of time steps

beta = pars['beta']
gamma = pars['gamma']
upsilon = pars['upsilon']
N = pars['N']
T = pars['T']
p_dist = pars['p_dist']
tau = T/N

for periodic in (True, False):
    mat = matrices.Matrices(tau, beta, gamma, p_dist, N, upsilon, periodic=periodic)

    print mat.A.toarray()[:5,:5]
    print mat.A.toarray()[-5:,-5:]

    try:
        assert (abs(mat.A.toarray()[:,:] - mat.A.toarray()[-1::-1,-1::-1]) < 1.e-10).all()
    except AssertionError:
        print "Matrix A not symmetric"
        print mat.A.toarray()[:,:] - mat.A.toarray()[-1::-1,-1::-1]

    print mat.B.toarray()[:5,:5]
    print mat.B.toarray()[-5:,-5:]

    try:
        assert (abs(mat.B.toarray()[:,:] - mat.B.toarray()[-1::-1,-1::-1]) < 1.e-10).all()
    except AssertionError:
        print "Matrix B not symmetric"
        print mat.B.toarray()[:,:] - mat.B.toarray()[-1::-1,-1::-1]