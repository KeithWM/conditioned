import scipy
import scipy.sparse

class Matrices(object):
    def __init__(self, tau, gamma, p_dist, N, upsilon=None, compute_expm=False, periodic=True):
        self.I = scipy.sparse.identity(N)

        diags = scipy.zeros((5,N)) # tridiagonal matrix for solving q evo
        # first, ignoring boundaries
        diags[-2,:] =  1
        diags[-1,:] = -4
        diags[0,:]  =  6
        diags[1,:]  = -4
        diags[2,:]  =  1
        if not periodic:
            # then the boundaries
            diags[0,0] = 1
            diags[1,1] = -2
            diags[2,2] = 1

            diags[-1,0] = -2
            diags[0,1]  = 5
            diags[1,2]  = -4
            diags[2,3]  = 1

            diags[1,-1]  = -2
            diags[0,-2]  = 5
            diags[-1,-3] = -4
            diags[-2,-4]  = 1

            diags[0,-1]  = 1
            diags[-1,-2] = -2
            diags[-2,-3] = 1

        diags*= -1/(tau**4)

        if periodic:
            diags[0,:] -= (gamma/tau)**2
            diags[2,:] += .5*(gamma/tau)**2
            diags[-2,:]+= .5*(gamma/tau)**2
        else:
            diags[0,2:-2] += (gamma/tau)**2
            diags[2,4:]   -= .5*(gamma/tau)**2
            diags[-2,:-4] -= .5*(gamma/tau)**2

            diags[0,0]+= gamma/tau**3 + (gamma/(2*tau))**2
            diags[1,1]-= gamma/tau**3
            diags[2,2]-= (gamma/(2*tau))**2

            diags[-1,0]-= gamma/tau**3
            diags[0,1] += gamma/tau**3

            diags[0,-2] += gamma/tau**3
            diags[1,-1] -= gamma/tau**3

            diags[0,-1] += gamma/tau**3 + (gamma/(2*tau))**2
            diags[-1,-2]-= gamma/tau**3
            diags[-2,-3]-= (gamma/(2*tau))**2

        # diags[-1,0] -= beta/(tau**3*(tau+p_dist))
        # diags[0,0]  += beta/(tau**3*(tau+p_dist))
        # diags[0,1]  += beta/(tau**3*(tau+p_dist))
        # diags[1,1]  -= beta/(tau**3*(tau+p_dist))
        self.A = scipy.sparse.spdiags(diags, [0,1,2,-2,-1], N, N).tolil()
        Afull = self.A.toarray()
        if periodic:
            self.A[0,-1] =  4/(tau**4)
            self.A[0,-2] = -1/(tau**4) + .5*(gamma/tau)**2
            self.A[1,-1] = -1/(tau**4) + .5*(gamma/tau)**2
            self.A[-1,0] =  4/(tau**4)
            self.A[-1,1] = -1/(tau**4) + .5*(gamma/tau)**2
            self.A[-2,0] = -1/(tau**4) + .5*(gamma/tau)**2
        self.A.tocsr()
        Afull = self.A.toarray()

        diags = scipy.zeros((3,N))
        if periodic:
            diags[-1,:] = 1
            diags[0,:]  = -2
            diags[1,:]  = 1
            diags*= 2/tau**2
        else:
            diags[-1,:-2] = 1
            diags[0,1:-1] = -2
            diags[1,2:]   = 1
            diags*= 2/tau**2

            diags[1,2] -=gamma/(2*tau)
            diags[-1,0]+=gamma/(2*tau)

            diags[1,-1]+=gamma/(2*tau)
            diags[-1,-3]-=gamma/(2*tau)
        self.B = scipy.sparse.spdiags(diags, [0,1,-1], N, N).tolil()

        if periodic:
            self.B[0,-1] = 2/tau**2
            self.B[-1,0] = 2/tau**2

        self.B.tocsc()

        if compute_expm == True:
            self.expmA = scipy.linalg.expm(-upsilon*self.A)
            self.expmB = scipy.linalg.expm(-upsilon*self.B)