import disk128 as disk
import numpy as np
from   scipy.linalg      import eig

# The eigenvalues should be the zeros of the integer-order Bessel functions.

def eigensystem(N,m,cutoff=1e9,report_error=False):
    
    z,w = disk.quadrature(N,niter=3,report_error=report_error)

    Q = disk.polynomial(N,0,m,z)

    E0 = disk.operator('E',N,0,m)
    E1 = disk.operator('E',N,1,m)
    D0 = disk.operator('D+',N,0,m)
    D1 = disk.operator('D-',N,1,m+1)

    R = E1.dot(E0)
    L = 2*D1.dot(D0)
    L,R = L.todense(), R.todense()
    L[N]=disk.operator('r=1',N,0,m)
    R[N]=np.zeros(N+1)

    vals, vecs = eig(L,b=-R)
    bad = (np.abs(vals) > cutoff)
    vals[bad] = np.nan
    vecs = vecs[:,np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]
    
    i = np.argsort(vals.real)
    vals, vecs = vals.real[i], vecs.real.transpose()[i].dot(Q)

    return vals, np.sqrt(0.5*(1+z)), vecs
