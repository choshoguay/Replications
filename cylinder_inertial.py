import disk128 as disk
import numpy as np
import scipy.sparse      as sparse
from   scipy.linalg      import eig

# The eigenvalues should be the same from the disk paper.

def eigensystem(N,m,alpha,cutoff=1e9,report_error=False):

    z,w = disk.quadrature(N,niter=3,report_error=report_error)

    Q = {'0':disk.polynomial(N,0,m,z)}
    Q['-'] = disk.polynomial(N,0,m-1,z)
    Q['+'] = disk.polynomial(N,0,m+1,z)

    E = {'0':(alpha**2)*disk.operator('E',N,0,m)}
    E['+'] = disk.operator('E',N,0,m+1)
    E['-'] = disk.operator('E',N,0,m-1)

    D = {'+0':disk.operator('D+',N,0,m)}
    D['-0'] = disk.operator('D-',N,0,m)
    D['+-'] = disk.operator('D+',N,0,m-1)
    D['-+'] = disk.operator('D-',N,0,m+1)

    u = np.sqrt(0.5)
    Z = disk.operator('0',N,0,m)
    
    L = sparse.bmat([ [ E['+'],Z,D['+0'] ], [ Z,-E['-'],D['-0']], [ Z,      Z,      E['0'] ] ])
    R = sparse.bmat([ [ E['+'],Z, Z      ], [ Z, E['-'],Z      ], [ D['-+'],D['+-'],Z      ] ])

    L,R = L.todense(), R.todense()
    
    L[N]=np.concatenate((u*disk.operator('r=1',N,0,m+1), u*disk.operator('r=1',N,0,m-1), np.zeros(N+1)))
    R[N]=np.zeros(3*(N+1))

    vals, vecs = eig(L,b=-R)
    bad = (np.abs(vals) > cutoff)
    vals[bad] = np.nan
    vecs = vecs[:,np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]
    
    i = np.argsort(vals.real)
    vals, vecs = vals.real[i], vecs.real.transpose()[i]

    N0,N1,N2=1*(N+1)-1,2*(N+1)-1,3*(N+1)-1

    vr  = u*(vecs[:,0:N0+1].dot(Q['+']) + vecs[:,N0+1:N1+1].dot(Q['-']))
    vth = u*(vecs[:,0:N0+1].dot(Q['+']) - vecs[:,N0+1:N1+1].dot(Q['-']))
    p   = vecs[:,N1+1:N2+1].dot(Q['0'])

    return vals, np.sqrt(0.5*(1+z)), vr, vth, p



