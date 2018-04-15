import disk128 as disk
import numpy as np
import scipy.sparse      as sparse
from   scipy.linalg      import eig

# The eigenvalues should be the same as the pipe-flow paper.

def eigensystem(N,m,nu,alpha,cutoff=1e9,report_error=False):

    def D(s,i,j):
        if s == +1: return disk.operator('D+',N,i,m+j)
        if s == -1: return disk.operator('D-',N,i,m+j)
    
    def E(i,j): return disk.operator('E',N,i,m+j)
    
    z,w = disk.quadrature(N,niter=3,report_error=report_error)

    Q = {'0':disk.polynomial(N,0,m,z),'-':disk.polynomial(N,0,m-1,z),'+':disk.polynomial(N,0,m+1,z)}

    N0,N1,N2,N3 = 1*(N+1)-1,2*(N+1)-1,3*(N+1)-1, 4*(N+1)-1
    
    e = np.sqrt(0.5)
    Z = disk.operator('0',N,0,m)

    R00 = E(1,+1).dot(E(0,+1))
    R11 = E(1,-1).dot(E(0,-1))
    R22 = E(1, 0).dot(E(0, 0))
    
    R=sparse.bmat([[R00,Z,Z,Z],[Z,R11,Z,Z],[Z,Z,R22,Z],[Z,Z,Z,Z]])
    R = R.todense()
    R[N0]=np.zeros(4*(N+1))
    R[N1]=np.zeros(4*(N+1))
    R[N2]=np.zeros(4*(N+1))
    
    L00 = nu*(2*D(+1,1,0).dot(D(-1,0,+1)) - alpha**2 * R00)
    L00 = 0.5*1j*alpha*( R00 - R00.dot( disk.operator('Z',N,0,m+1) ) ) - L00
    
    
    L11 = nu*(2*D(-1,1,0).dot(D(+1,0,-1)) - alpha**2 * R11)
    L11 = 0.5*1j*alpha*( R11 - R11.dot( disk.operator('Z',N,0,m-1) ) ) - L11
    
    
    L22 = nu*(2*D(-1,1,+1).dot(D(+1,0,0)) - alpha**2 * R22)
    L22 = 0.5*1j*alpha*( R22 - R22.dot( disk.operator('Z',N,0,m) ) )   - L22
    
    
    L03 = E(+1,+1).dot(D(+1,0,0))
    L13 = E(+1,-1).dot(D(-1,0,0))
    L23 = 1j*alpha*R22
    
    L30 = D(-1,0,+1)
    L31 = D(+1,0,-1)
    L32 = 1j*alpha*E(0,0)
    
    L20 = -2*e*R22.dot(disk.operator('R-',N,0,m+1))
    L21 = -2*e*R22.dot(disk.operator('R+',N,0,m-1))
    
    
    L=sparse.bmat([[L00,Z,Z,L03], [Z,L11,Z,L13],[L20,L21,L22,L23],[L30,L31,L32,Z]])
    L = L.todense()
    L[N0]=np.concatenate((e*disk.operator('r=1',N,0,m+1), e*disk.operator('r=1',N,0,m-1),np.zeros(N1+1)))
    L[N1]=np.concatenate((e*disk.operator('r=1',N,0,m+1),-e*disk.operator('r=1',N,0,m-1),np.zeros(N1+1)))
    L[N2]=np.concatenate((np.zeros(N1+1),disk.operator('r=1',N,0,m  ),np.zeros(N0+1)))
    
    # The rate-limiting step
    vals, vecs = eig(L,b=-R)
    bad = (np.abs(vals) > cutoff)
    vals[bad] = np.nan
    vecs = vecs[:,np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]
    
    i = np.argsort(-vals.real)
    vals, vecs = vals[i], vecs.transpose()[i]

    u = e*(vecs[:,0:N0+1].dot(Q['+'])+vecs[:,N0+1:N1+1].dot(Q['-']))
    v = e*(vecs[:,0:N0+1].dot(Q['+'])-vecs[:,N0+1:N1+1].dot(Q['-']))
    w = vecs[:,N1+1:N2+1].dot(Q['0'])
    p = vecs[:,N2+1:N3+1].dot(Q['0'])
    
    return vals, np.sqrt(0.5*(1+z)), u, v, w, p





