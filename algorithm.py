"""
See Overleaf for the document describing the ideas:
https://www.overleaf.com/6439819vrtbrz
"""

import scipy
import scipy.fftpack
import matrices
from scipy.sparse.linalg import spsolve

class Algorithm(object):
    """
    Original implementation with spectral noise, not corrected for the finite physical time step, i.e. I assume the
    physical time discretization is exactly equal to the continuous, second-order, physical time derivative.
    """
    def __init__(self, pot, pars):
        self.generic_initialization(pot, pars)

        d2_complex = -(2*scipy.pi/self.T*scipy.arange(self.N/2+1))**2
        # d2 = scipy.zeros((self.N,))
        # d2[0::2] = d2_complex[:-1]
        # d2[1::2] = d2_complex[1:]
        d2 = self.fft_to_rfft(d2_complex)
        self.d2 = d2
        self.d4 = d2**2
        self.theta = (self.d4 - self.gamma**2*d2)

        self.ou1 = scipy.exp( -self.upsilon_linear_nonlocal*self.theta )
        # ou2 = scipy.sqrt( 1*gamma*N*(1-ou1**2)/(beta*tau) )/scipy.sqrt(theta)
        self.ou2 = scipy.sqrt( self.sigma*2*self.gamma*self.N*(1-self.ou1**2)/(2*self.beta*self.theta*self.tau) )
        self.ou2[0] = scipy.sqrt( self.sigma*4*self.gamma*self.N*self.upsilon_linear_nonlocal/(self.beta*self.tau) )
        self.ou2[-1]*= scipy.sqrt(3)

        self.ou_four = scipy.sqrt( self.sigma*2*self.gamma*self.N*self.upsilon_linear_nonlocal/(self.beta*self.tau))*scipy.ones((self.N,) )
        self.ou_four[0]*= scipy.sqrt(2)
        self.ou_four[-1]*= scipy.sqrt(3)
        self.ou_phys = scipy.sqrt( self.sigma*4*self.gamma*self.upsilon_linear_nonlocal/(self.beta*self.tau) )

    def fft_to_rfft(self, input):
        assert input.shape[0] == self.N/2+1
        output = scipy.zeros((self.N,))
        output[0::2] = input[:-1]
        output[1::2] = input[1:]
        return output


    def generic_initialization(self, pot, pars):
        self.pars = pars
        self.beta = pars['beta']
        self.gamma = pars['gamma']
        self.upsilon = pars['upsilon']
        self.Strang_order = pars['Strang_order']
        self.N = pars['N']
        self.T = pars['T']
        self.sigma = pars['sigma']

        self.tau = self.T/self.N # time step
        self.ts = scipy.arange(0, self.N)*self.tau

        self.pot = pot

        self.gL = +1.
        self.gR = -1

        self.qL = self.gL # these are the trivial inversions of g...
        self.qR = self.gR

        self.Strang_functions = (self.update_linear_nonlocal, self.update_nonlinear_nonlocal, self.update_nonlinear_local)
        self.upsilon_linear_nonlocal    = 0.
        self.upsilon_nonlinear_nonlocal = 0.
        self.upsilon_nonlinear_local    = 0.
        if not (self.Strang_order==0).sum() == 0:
            self.upsilon_linear_nonlocal    = self.upsilon/(self.Strang_order==0).sum()
        if not (self.Strang_order==1).sum() == 0:
            self.upsilon_nonlinear_nonlocal = self.upsilon/(self.Strang_order==1).sum()
        if not (self.Strang_order==2).sum() == 0:
            self.upsilon_nonlinear_local    = self.upsilon/(self.Strang_order==2).sum()
        self.upsilons = (self.upsilon_linear_nonlocal, self.upsilon_nonlinear_nonlocal, self.upsilon_nonlinear_local)


    def g(self, q):
        return q

    def dg(self, q):
        return 1

    def transform(self, qs, qs_hat):
        qs_hat[:] = scipy.fftpack.rfft(qs)

    def inverse_transform(self, qs, qs_hat):
        qs[:] = scipy.fftpack.irfft(qs_hat)

    def constrain(self, qs):
        qs[self.N/4]   = self.gL
        qs[3*self.N/4] = self.gR
        return

    def update_linear_nonlocal(self, qs, qs_hat, fs, fs_hat):
        self.transform(qs, qs_hat)
        qs_hat[:] = self.ou1*qs_hat + self.ou2*scipy.random.normal(size=(self.N,))
        self.inverse_transform(qs, qs_hat)
        return

    def update_nonlinear_nonlocal(self, qs, qs_hat, fs, fs_hat):
        self.transform(qs, qs_hat)
        fs_hat[:] = qs_hat*self.d2
        self.inverse_transform(qs, qs_hat)
        self.inverse_transform(fs, fs_hat)
        fs*= -2*self.pot.ddV(qs)
        # plt.plot(ts, fs,':')
        # plt.show()
        qs += fs * self.upsilon_nonlinear_nonlocal
        return

    def update_nonlinear_local(self, qs, qs_hat, fs, fs_hat):
        dVddV = scipy.zeros_like(qs) # only the terms like V'(q)V''(q)
        dVddV[1:-1] -= .5*self.pot.dV(qs)[1:-1]*self.pot.ddV(qs)[1:-1]
        qs += dVddV * self.upsilon_nonlinear_local
        return

    def update(self, qs, qs_hat, fs, fs_hat):
        for function_i in self.Strang_order:
            self.Strang_functions[function_i](qs, qs_hat, fs, fs_hat)
        # self.constrain(qs)

    def iterate(self, q0, alg_times):
        qs = scipy.array(q0) # make sure to copy the values
        qs_hat = scipy.zeros((self.N,))
        fs = scipy.zeros_like(q0) # an array to store a force
        fs_hat = scipy.zeros((self.N,))

        qs_save = scipy.zeros(alg_times.shape + q0.shape)

        qs_save[0,:] = qs

        s = 0
        for u, alg_time in enumerate(alg_times[:-1]):
            while s < alg_times[u+1]:
                self.update(qs, qs_hat, fs, fs_hat)
                s+= self.upsilon
            print '{}'.format(u),

            qs_save[u+1,:] = qs

        print
        return qs_save


class AlgorithmQuasiPseudospectralSemiImplicit(Algorithm):
    """
    Here the equations are not changed to before, but the noise is applied in physical time space, not Fourier space and
    a semi-implicit scheme is used for this (see T. Li, A. Abdulle and W. E / Commun. Comput. Phys., 3 (2008))
    """
    def update_linear_nonlocal(self, qs, qs_hat, fs, fs_hat):
        self.transform(qs, qs_hat)
        qs_hat[:] -= self.theta*qs_hat*.5*self.upsilon_linear_nonlocal
        self.inverse_transform(qs, qs_hat)
        qs += self.ou_phys*scipy.random.normal(size=(self.N,))
        self.transform(qs, qs_hat)
        qs_hat[:] = qs_hat/(1 + self.theta*.5*self.upsilon_linear_nonlocal)
        self.inverse_transform(qs, qs_hat)
        return


class AlgorithmPseudospectralSemiImplicit(Algorithm):
    """
    Using Fourier noise in a semi-implicit scheme (see T. Li, A. Abdulle and W. E / Commun. Comput. Phys., 3 (2008))
    """
    def update_linear_nonlocal(self, qs, qs_hat, fs, fs_hat):
        self.transform(qs, qs_hat)
        qs_hat[:] -= self.theta*qs_hat*.5*self.upsilon_linear_nonlocal
        qs_hat += self.ou_four*scipy.random.normal(size=(self.N,))
        qs_hat[:] = qs_hat/(1 + self.theta*.5*self.upsilon_linear_nonlocal)
        self.inverse_transform(qs, qs_hat)
        return


class AlgorithmSemiImplicit(Algorithm):
    """
    Without transforming to Fourier space at all
    """

    def __init__(self, pot, pars):

        self.generic_initialization(pot, pars)
        self.p_dist = pars['p_dist']
        self.mat = matrices.Matrices(self.tau, self.gamma, self.p_dist, self.N, periodic=pars['periodic'])
        self.ou_phys = scipy.sqrt(self.sigma* 4*self.gamma*self.upsilon_linear_nonlocal/(self.beta*self.tau))

        self.rhs_matrix = scipy.sparse.identity(self.N) + .5*self.mat.A*self.upsilon_linear_nonlocal
        self.lhs_matrix = scipy.sparse.identity(self.N) - .5*self.mat.A*self.upsilon_linear_nonlocal



    def update_linear_nonlocal(self, qs):
        dW = self.ou_phys * scipy.random.normal(size=(self.N,))
        qs[:] = spsolve(self.lhs_matrix, self.rhs_matrix*qs + dW)
        return

    def update_nonlinear_nonlocal(self, qs):
        qs -=  2*self.pot.ddV(qs)*self.mat.B*qs * self.upsilon_nonlinear_nonlocal
        return

    def update_nonlinear_local(self, qs):
        dVddV = scipy.zeros_like(qs) # only the terms like V'(q)V''(q)
        dVddV[1:-1] -= .5*self.pot.dV(qs)[1:-1]*self.pot.ddV(qs)[1:-1]
        qs += dVddV * self.upsilon_nonlinear_local
        return

    def update(self, qs):
        self.constrain(qs)
        for function_i in self.Strang_order:
            self.Strang_functions[function_i](qs)

    def iterate(self, q0, alg_times):
        qs = scipy.array(q0) # make sure to copy the values
        fs = scipy.zeros_like(q0) # an array to store a force

        qs_save = scipy.zeros(alg_times.shape + q0.shape)

        qs_save[0,:] = qs
        s = 0
        for u, alg_time in enumerate(alg_times[:-1]):
            while s < alg_times[u+1]:
                self.update(qs)
                s+= self.upsilon

            qs_save[u+1,:] = qs

        return qs_save


class AlgorithmInversePseudospectralSemiImplicit(Algorithm):
    """
    Applying the noise spectrally, but updatign the stiff linear part in physical time space
    """

    def __init__(self, pot, pars):

        self.generic_initialization(pot, pars)

        self.ou_four = scipy.sqrt(self.sigma*2*self.gamma*self.N*self.upsilon_linear_nonlocal/(self.beta*self.tau))

        self.p_dist = pars['p_dist']
        self.mat = matrices.Matrices(self.tau, self.gamma, self.p_dist, self.N, periodic=pars['periodic'])

        self.rhs_matrix = scipy.sparse.identity(self.N) + .5*self.mat.A*self.upsilon_linear_nonlocal
        self.lhs_matrix = scipy.sparse.identity(self.N) - .5*self.mat.A*self.upsilon_linear_nonlocal

    def update_linear_nonlocal(self, qs, qs_hat, fs, fs_hat):
        qs[:] = self.rhs_matrix*qs
        self.transform(qs, qs_hat)
        qs_hat += self.ou_four*scipy.random.normal(size=(self.N,))
        self.inverse_transform(qs, qs_hat)
        qs[:] = spsolve(self.lhs_matrix, qs)
        return


class AlgorithmTaylor(Algorithm):
    """
    As the original (spectral space for noise AND linear, non_local part), but now the spectral transformation for the
    linear part takes into account the error in the approximation of the linear stencil/matrix as an approximation for a
    derivative. Due to the spectral noise, the derivatives d^k/dt^k q in physical time do not vanish for large k, so we
    take all of them into account. The sum of all the linear derivative operators in spectral space amounts to a sine
    popping up in the computation of d2 and d4.
    """
    def __init__(self, pot, pars):
        self.generic_initialization(pot, pars)

        ks = scipy.arange(0, pars['N']/2+1, dtype=int)

        d2_complex = -(scipy.sin((2*scipy.pi*ks)/pars['N']) / self.tau)**2
        d2 = scipy.zeros((self.N,))
        d2[0::2] = d2_complex[:-1]
        d2[1::2] = d2_complex[1:]

        d4_complex = 16*(scipy.sin((scipy.pi*ks)/pars['N']) / self.tau)**4
        d4 = scipy.zeros((self.N,))
        d4[0::2] = d4_complex[:-1]
        d4[1::2] = d4_complex[1:]
        self.d2 = d2
        self.d4 = d4
        self.theta = (self.d4 - self.gamma**2*d2)

        self.ou1 = scipy.exp( -self.upsilon_linear_nonlocal*self.theta )
        # ou2 = scipy.sqrt( 1*gamma*N*(1-ou1**2)/(beta*tau) )/scipy.sqrt(theta)
        self.ou2 = scipy.sqrt( self.sigma*2*self.gamma*self.N*(1-self.ou1**2)/(2*self.beta*self.theta*self.tau) )
        self.ou2[0] = scipy.sqrt( self.sigma*4*self.gamma*self.N*self.upsilon_linear_nonlocal/(self.beta*self.tau) )
        self.ou2[-1]*= scipy.sqrt(3) # due to the eigenaardigheden of spectral noise...


class AlgorithmTaylorBAOAB(Algorithm):
    """
    As AlgorithmTaylor, but now starting from the BAOAB time discretization scheme
    """
    def __init__(self, pot, pars):
        self.generic_initialization(pot, pars)

        ks = scipy.arange(0, pars['N']/2+1, dtype=int)

        d2_complex = -(scipy.sin((2*scipy.pi*ks)/pars['N']) / self.tau)**2
        d2 = scipy.zeros((self.N,))
        d2[0::2] = d2_complex[:-1]
        d2[1::2] = d2_complex[1:]

        d4_complex = 16*(scipy.sin((scipy.pi*ks)/pars['N']) / self.tau)**4
        d4 = scipy.zeros((self.N,))
        d4[0::2] = d4_complex[:-1]
        d4[1::2] = d4_complex[1:]
        self.d2 = d2
        self.d4 = d4
        self.theta = (scipy.cos(.5*self.gamma*self.tau)*self.d4 - scipy.sin(.5*self.gamma*self.tau)*2/self.tau*self.gamma**2*d2)

        self.ou1 = scipy.exp( -self.upsilon_linear_nonlocal*self.theta )
        # ou2 = scipy.sqrt( 1*gamma*N*(1-ou1**2)/(beta*tau) )/scipy.sqrt(theta)
        self.ou2 = scipy.sqrt( self.sigma*2*self.gamma*self.N*(1-self.ou1**2)/(2*self.beta*self.theta*self.tau) )
        self.ou2[0] = scipy.sqrt( self.sigma*4*self.gamma*self.N*self.upsilon_linear_nonlocal/(self.beta*self.tau) )
        self.ou2[-1]*= scipy.sqrt(3) # due to the eigenaardigheden of spectral noise...

class AlgorithmOUConstraints(Algorithm):
    """
    In this approach, the idea is that OU process itself satisfies the constraints. It seems that otherwise the
    imposing the constraints heavily disrupts the solution.
    The matrix self.PI is constructed to be the projection operator onto the cotangent space (see *).
    This approach seems to be more involved than I anticipated. As each degree of freedom has a different friction, it is
    not possible to apply the methods of Leimkuhler and Matthews 2016 out of the box.
    * On second inspection, I no longer see why LeMa16 would be relevant, that is Hamiltonian, we are overdamped.
    """
    def __init__(self, pot, pars):
        self.generic_initialization(pot, pars)

        ks = scipy.arange(0, pars['N']/2+1, dtype=int)

        d2_complex = -(scipy.sin((2*scipy.pi*ks)/pars['N']) / self.tau)**2
        d2 = self.fft_to_rfft(d2_complex)

        d4_complex = 16*(scipy.sin((scipy.pi*ks)/pars['N']) / self.tau)**4
        d4 = self.fft_to_rfft(d4_complex)
        self.d2 = d2
        self.d4 = d4
        self.theta = (self.d4 - self.gamma**2*d2)

        self.ou1 = scipy.exp( -self.upsilon_linear_nonlocal*self.theta )
        # ou2 = scipy.sqrt( 1*gamma*N*(1-ou1**2)/(beta*tau) )/scipy.sqrt(theta)
        self.ou2 = scipy.sqrt( self.sigma*2*self.gamma*self.N*(1-self.ou1**2)/(2*self.beta*self.theta*self.tau) )
        self.ou2[0] = scipy.sqrt( self.sigma*4*self.gamma*self.N*self.upsilon_linear_nonlocal/(self.beta*self.tau) )
        self.ou2[-1]*= scipy.sqrt(3) # due to the eigenaardigheden of spectral noise...

        self.constrained = self.pars['constrained']
        self.constraints = self.pars['constraints']
        self.N_constraints = len(self.constrained)

        G = scipy.zeros((self.N_constraints, self.N))# constraint matrix (in Fourier space)
        for n_constraint, j in enumerate(self.constrained):
            complex_coeff = scipy.exp(( 2*scipy.pi*scipy.sqrt(-1)*j*scipy.arange(self.N/2) )/self.N)/self.N
            G[n_constraint, 0::2] = scipy.real(complex_coeff)
            G[n_constraint, 1::2] = scipy.imag(complex_coeff)
        M_diag = scipy.zeros((self.N,))
        M_diag[0]    = self.sigma*4*self.gamma*self.N / (self.beta*self.tau)
        M_diag[1:-1] = self.sigma*2*self.gamma*self.N / (self.beta*self.tau)
        M_diag[-1]   = self.sigma*6*self.gamma*self.N / (self.beta*self.tau)
        M = scipy.diag(M_diag)
        M_inv = scipy.linalg.inv(M)
        GM_inv = scipy.dot(G, M_inv)
        inner = scipy.dot( GM_inv, G.T)
        inner_inv = scipy.linalg.inv(inner)
        self.PI = scipy.identity(self.N) - scipy.dot( scipy.dot(G.T, inner_inv), GM_inv)

    def update_linear_nonlocal(self, qs, qs_hat, fs, fs_hat):
        self.transform(qs, qs_hat)
        qs_hat[:] = self.ou1*qs_hat + self.ou2*scipy.dot(self.PI, scipy.random.normal(size=(self.N,)))
        self.inverse_transform(qs, qs_hat)
        return

class AlgorithmOUConstraints(Algorithm):
    """
    In this approach, the idea is that OU process itself satisfies the constraints. It seems that otherwise the
    imposing the constraints heavily disrupts the solution.
    The matrix self.PI is constructed to be the projection operator onto the cotangent space (see *).
    This approach seems to be more involved than I anticipated. As each degree of freedom has a different friction, it is
    not possible to apply the methods of Leimkuhler and Matthews 2016 out of the box.
    * On second inspection, I no longer see why LeMa16 would be relevant, that is Hamiltonian, we are overdamped.
    """
    def __init__(self, pot, pars):
        self.generic_initialization(pot, pars)

        ks = scipy.arange(0, pars['N']/2+1, dtype=int)

        d2_complex = -(scipy.sin((2*scipy.pi*ks)/pars['N']) / self.tau)**2
        d2 = self.fft_to_rfft(d2_complex)

        d4_complex = 16*(scipy.sin((scipy.pi*ks)/pars['N']) / self.tau)**4
        d4 = self.fft_to_rfft(d4_complex)
        self.d2 = d2
        self.d4 = d4
        self.theta = (self.d4 - self.gamma**2*d2)

        self.ou1 = scipy.exp( -self.upsilon_linear_nonlocal*self.theta )
        # ou2 = scipy.sqrt( 1*gamma*N*(1-ou1**2)/(beta*tau) )/scipy.sqrt(theta)
        self.ou2 = scipy.sqrt( self.sigma*2*self.gamma*self.N*(1-self.ou1**2)/(2*self.beta*self.theta*self.tau) )
        self.ou2[0] = scipy.sqrt( self.sigma*4*self.gamma*self.N*self.upsilon_linear_nonlocal/(self.beta*self.tau) )
        self.ou2[-1]*= scipy.sqrt(3) # due to the eigenaardigheden of spectral noise...

        self.constrained = self.pars['constrained']
        self.constraints = self.pars['constraints']
        self.N_constraints = len(self.constrained)

        G = scipy.zeros((self.N_constraints, self.N))# constraint matrix (in Fourier space)
        for n_constraint, j in enumerate(self.constrained):
            complex_coeff = scipy.exp(( 2*scipy.pi*scipy.sqrt(-1)*j*scipy.arange(self.N/2) )/self.N)/self.N
            G[n_constraint, 0::2] = scipy.real(complex_coeff)
            G[n_constraint, 1::2] = scipy.imag(complex_coeff)
        M_diag = scipy.zeros((self.N,))
        M_diag[0]    = self.sigma*4*self.gamma*self.N / (self.beta*self.tau)
        M_diag[1:-1] = self.sigma*2*self.gamma*self.N / (self.beta*self.tau)
        M_diag[-1]   = self.sigma*6*self.gamma*self.N / (self.beta*self.tau)
        M = scipy.diag(M_diag)
        M_inv = scipy.linalg.inv(M)
        GM_inv = scipy.dot(G, M_inv)
        inner = scipy.dot( GM_inv, G.T)
        inner_inv = scipy.linalg.inv(inner)
        self.PI = scipy.identity(self.N) - scipy.dot( scipy.dot(G.T, inner_inv), GM_inv)

    def update_linear_nonlocal(self, qs, qs_hat, fs, fs_hat):
        self.transform(qs, qs_hat)
        qs_hat[:] = self.ou1*qs_hat + self.ou2*scipy.dot(self.PI, scipy.random.normal(size=(self.N,)))
        self.inverse_transform(qs, qs_hat)
        return

class AlgorithmOUConstraintsHamiltonian(Algorithm):
    """
    In this approach, we use a Langevin perturbed Hamiltonian dynamics to sample the desired probability distribution over
    the phase space, where the Langevin perturbation itself satisfies the constraints, see Leimkuhler and Matthews 2016
    The matrix self.PI is constructed to be the projection operator onto the cotangent space (see *).
    """

    def __init__(self, pot, pars):
        self.generic_initialization(pot, pars)

        self.chi = pars['chi'] # the friction to be used for the Langevin dynamics in pseudo-time

        ks = scipy.arange(0, pars['N']/2+1, dtype=int)

        d2_complex = -(scipy.sin((2*scipy.pi*ks)/pars['N']) / self.tau)**2
        d2 = scipy.zeros((self.N,))
        d2[0::2] = d2_complex[:-1]
        d2[1::2] = d2_complex[1:]

        d4_complex = 16*(scipy.sin((scipy.pi*ks)/pars['N']) / self.tau)**4
        d4 = scipy.zeros((self.N,))
        d4[0::2] = d4_complex[:-1]
        d4[1::2] = d4_complex[1:]
        self.d2 = d2
        self.d4 = d4
        self.theta = (self.d4 - self.gamma**2*d2)

        self.sinterm = scipy.sin(self.theta * self.upsilon/2)
        self.costerm = scipy.cos(self.theta * self.upsilon/2)

        self.ou1 = scipy.exp( -self.upsilon_linear_nonlocal*self.theta )
        # ou2 = scipy.sqrt( 1*gamma*N*(1-ou1**2)/(beta*tau) )/scipy.sqrt(theta)
        self.ou2 = scipy.sqrt( self.sigma*2*self.gamma*self.N*(1-self.ou1**2)/(2*self.beta*self.theta*self.tau) )
        self.ou2[0] = scipy.sqrt( self.sigma*4*self.gamma*self.N*self.upsilon_linear_nonlocal/(self.beta*self.tau) )
        self.ou2[-1]*= scipy.sqrt(3) # due to the eigenaardigheden of spectral noise...


        self.constrained = self.pars['constrained']
        self.constraints = self.pars['constraints']
        self.N_constraints = len(self.constrained)

        G = scipy.zeros((self.N_constraints, self.N))  # constraint matrix (in Fourier space)
        for n_constraint, j in enumerate(self.constrained):
            complex_coeff = scipy.exp(
                (2 * scipy.pi * scipy.sqrt(-1) * j * scipy.arange(self.N / 2)) / self.N) / self.N
            G[n_constraint, 0::2] = scipy.real(complex_coeff)
            G[n_constraint, 1::2] = scipy.imag(complex_coeff)
        M_diag = scipy.zeros((self.N,))
        M_diag[0] = self.sigma * 4 * self.gamma * self.N / (self.beta * self.tau)
        M_diag[1:-1] = self.sigma * 2 * self.gamma * self.N / (self.beta * self.tau)
        M_diag[-1] = self.sigma * 6 * self.gamma * self.N / (self.beta * self.tau)
        M = scipy.diag(M_diag)
        M_inv = scipy.linalg.inv(M)
        GM_inv = scipy.dot(G, M_inv)
        inner = scipy.dot(GM_inv, G.T)
        inner_inv = scipy.linalg.inv(inner)
        self.PI = scipy.identity(self.N) - scipy.dot(scipy.dot(G.T, inner_inv), GM_inv)

    def ABg_linear_nonlocal(self,qs, qs_hat, ps, ps_hat, fs, fs_hat, dt):
        self.transform(qs, qs_hat)
        self.transform(ps, ps_hat)
        # ps_hat[:]-= dt * scipy.dot(self.PI, self.theta*qs_hat)
        # ps_hat[:], qs_hat[:] = ps_hat*self.costerm - qs_hat*self.sinterm, ps_hat*self.sinterm + qs_hat*self.costerm
        pstar     = ps_hat*self.costerm - qs_hat*self.sinterm
        qs_hat[:] = ps_hat*self.sinterm + qs_hat*self.costerm
        ps_hat[:] = pstar
        self.inverse_transform(qs, qs_hat)
        self.inverse_transform(ps, ps_hat)
        return

    def ABg(self, qs, qs_hat, ps, ps_hat, fs, fs_hat, dt):
        # the 'kick' momentum change due to the potential
        self.ABg_linear_nonlocal(qs, qs_hat, ps, ps_hat, fs, fs_hat, dt)

    def Og(self, qs, qs_hat, ps, ps_hat, fs, fs_hat):
        # the OU part of the process dp = -chi dt + sqrt(2 chi) dW
        return


    def update(self, qs, qs_hat, ps, ps_hat, fs, fs_hat):
        # self.constrain(qs)
        self.ABg(qs, qs_hat, ps, ps_hat, fs, fs_hat, self.upsilon/2) # B as in BAOAB
        # self.Ag(qs, qs_hat, ps, ps_hat, fs, fs_hat,  self.upsilon/2) # A ...
        # self.Og(qs, qs_hat, ps, ps_hat, fs, fs_hat) # O
        # self.Ag(qs, qs_hat, ps, ps_hat, fs, fs_hat, self.upsilon/2) # A
        self.ABg(qs, qs_hat, ps, ps_hat, fs, fs_hat, self.upsilon/2) # B

    def iterate(self, q0, alg_times):
        qs = scipy.array(q0)  # make sure to copy the values
        qs_hat = scipy.zeros((self.N,))
        ps = scipy.zeros_like(q0)  # an array to store momenta
        ps_hat = scipy.zeros((self.N,))
        fs = scipy.zeros_like(q0)  # an array to store a force
        fs_hat = scipy.zeros((self.N,))

        qs_save = scipy.zeros(alg_times.shape + q0.shape)

        qs_save[0, :] = qs

        s = 0
        for u, alg_time in enumerate(alg_times[:-1]):
            while s < alg_times[u + 1]:
                self.update(qs, qs_hat, ps, ps_hat, fs, fs_hat)
                s += self.upsilon
            print '{}'.format(u),

            qs_save[u + 1, :] = qs

        print
        return qs_save