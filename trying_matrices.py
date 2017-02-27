import scipy
import scipy.sparse
import scipy.linalg

N = 12
gamma = 1.
tau = .1

B = scipy.zeros((N+2,3))
B[:,1] = scipy.cos(.5*gamma*tau) + scipy.sin(.5*gamma*tau)/tau
B[:,-1] = scipy.cos(.5*gamma*tau) - scipy.sin(.5*gamma*tau)/tau
B[:,0] = -2*scipy.cos(.5*gamma*tau)
A = scipy.sparse.spdiags(B.T, [0,1,-1], N,N+1).toarray()
# A[0,-1] = A[1,0]
# A[-1,0] = A[0,1]
A[0,0] = -1
A[0,1] = 1
# Ainv = scipy.linalg.inv(A)
ATA = scipy.dot(A.T, A)

A2 = scipy.sparse.spdiags(B.T, [0,1,-1], N,N).toarray()
A2[0,0] = -1
A2[0,1] = -1
A2[-1,-1] = 1
A2[-1,-2] = -1
ATA2 = scipy.dot(A2.T, A2)

R2L = scipy.sparse.spdiags(scipy.ones((2,N+1), dtype=int), [0, -1], N, N+1).toarray()
# R2L[0,-1] = 1
R2L [0,-1] = 2 # for the initial distribution on p, '2' is a placeholder vlaue

S = scipy.dot(R2L, R2L.T)
Sinv = scipy.linalg.inv(S)

print scipy.linalg.eigvals(S)


SinvA = scipy.dot(Sinv, A)