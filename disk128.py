import numpy             as np
import scipy             as sp
import jacobi128         as jacobi

alpha = -1/2 # a default for a default

def quadrature(N_max,a=alpha,**kw):
    return jacobi.quadrature(N_max,a,0,**kw)

def polynomial(N_max,k,m,z,a=alpha):
    
    q = k + a
    
    N = N_max # - some function of m
    
    init = np.sqrt(2**k)*jacobi.envelope(q,m,q,0,z)
    
    return jacobi.recursion(N,q,m,z,init)

def unitary(rank=1,adjoint=False):

    if rank == 0: return 1
    
    if adjoint :
        U       = np.sqrt(0.5)*np.array([[1,1],[-1j,1j]])
    else:
        U       = np.sqrt(0.5)*np.array([[1,1j],[1,-1j]])
    unitary = U
    for k in range(rank-1):
        unitary = np.kron(U,unitary)

    return unitary

def operator(op,N_max,k,m,a=alpha):
    
    q = k + a
    
    N = N_max # - some function of m
    
    # null
    if (op == '0'):  return jacobi.operator('0',N,q,m)
    
    # identity
    if (op == 'I'):  return jacobi.operator('I',N,q,m)
    
    # conversion
    if (op == 'E'):  return jacobi.operator('A+',N,q,m,rescale=np.sqrt(0.5))
    
    # derivatives
    if (op == 'D-'): return jacobi.operator('C+',N,q,m,rescale=np.sqrt(2.0))
    if (op == 'D+'): return jacobi.operator('D+',N,q,m,rescale=np.sqrt(2.0))

    # r multiplication
    if (op == 'R-'): return jacobi.operator('B-',N,q,m,rescale=np.sqrt(0.5))
    if (op == 'R+'): return jacobi.operator('B+',N,q,m,rescale=np.sqrt(0.5))

    # z = 2*r*r-1 multiplication
    if op == 'Z': return jacobi.operator('J',N,q,m)

    if op == 'r=1': return jacobi.operator('z=+1',N,q,m,rescale=np.sqrt(0.5))



