import numpy as np
import cytnx as cy
from ncon import ncon
import scipy
from scipy.sparse.linalg import eigsh, eigs
from cmath import sqrt

def tonumpy(uniten):
    return uniten.get_block_().numpy()

def truncate(v):

    '''
    Padding small (i.e 1e-41) eigenvalues of a Hermiaian matrix with random reasonable number.
    '''

    D,U = np.linalg.eigh(v)
    for i in range(len(D)):
        if D[i]<0:
            D[i] *= -1 
        if D[i]<1e-15:
            D[i] = np.random.rand(1)[0] * 1e-15
    v_ = U @ np.diag(D) @ U.conj().T
    return v_

def getlr(A):
    A = tonumpy(A)
    m = A.shape[0]
    TM = ncon([A, A.conj()], [[-1,1,-3], [-2,1,-4]]).reshape(m**2, m**2)
    e, r = eigs(TM, k = 1, which = 'LM')
    e, l = eigs(TM.T, k = 1, which = 'LM') 
    r = r.reshape(m, m)
    l = l.reshape(m, m)
    r = 0.5*(r+r.conj().T)
    l = 0.5*(l+l.conj().T)
    r = truncate(r)
    l = truncate(l)
    tr = ncon([l,r],[[1,2],[1,2]])
    l/=tr**0.5
    r/=tr**0.5
    return cy.UniTensor(cy.from_numpy(l), 1), cy.UniTensor(cy.from_numpy(r), 1)

def getInverselr(l, r):
    l, r = tonumpy(l), tonumpy(r)
    return cy.UniTensor(cy.from_numpy(scipy.linalg.sqrtm(scipy.linalg.inv(l))), 1), cy.UniTensor(cy.from_numpy(scipy.linalg.inv(r)), 1)

def sqrtm(v):
    D, U = cy.linalg.Eigh(v.get_block())
    D_ = cy.zeros(len(D), dtype = cy.Type.ComplexDouble)
    for i in range(len(D)):
        D_[i] = sqrt(D[i].item())
    D_ = cy.linalg.Diag(D_)
    Ud = U.Conj().permute(1, 0)
    v_ = cy.linalg.Matmul(cy.linalg.Matmul(U, D_), Ud)
    return cy.UniTensor(v_, 0)

def LeftGaugefixed(A, l, r):
    A, l, r = tonumpy(A), tonumpy(l), tonumpy(r)
    m = A.shape[0]
    d = A.shape[1]
    temp = ncon([scipy.linalg.sqrtm(l), A.conj()],[[-2, 1],[1, -3, -1]]).reshape(m, m*d)
    Vl = scipy.linalg.null_space(temp)[:, range(m*(d-1))]
    Vl = np.linalg.svd(Vl, full_matrices=False)[0].reshape(m,d,m*(d-1))
    return cy.UniTensor(cy.from_numpy(Vl), 0)

def toSymmetricGauge(AL):

    def toLambdaGamma(AL):
        AL.set_rowrank(2)
        s, u, vt = cy.linalg.Svd(AL)
        vt.set_labels([-1,1]); u.set_labels([1,-2,-3])
        Gamma = cy.Contract(vt, u)
        Gamma.set_rowrank(0)
        Lambda = cy.UniTensor(cy.linalg.Diag(s.get_block()) / cy.linalg.Norm(s.get_block()).item(), 0)
        return Lambda, Gamma

    def toCanonical(Lambda, Gamma):

        '''
        https://arxiv.org/pdf/0711.3960.pdf
        '''

        def getTMlr(TM, m):
            e, v = eigs(TM, k = 1, which = 'LM')
            v = v.reshape(D, D)
            v = 0.5*(v+v.conj().T)
            return cy.UniTensor(cy.from_numpy(truncate(v)), 0)
            
        D = Lambda.shape()[0]

        Lnet = cy.Network()
        Lnet.FromString(["Lambda: ;-1,1", "Gamma: ;1,2,-3", "Lambda_: ;-2,3", "Gamma_conj: ;3,2,-4", "TOUT: ;-1,-2,-3,-4"])
        Rnet = cy.Network()
        Rnet.FromString(["Gamma: ;-1,2,1", "Lambda: ;1,-3", "Gamma_conj: ;-2,2,3", "Lambda_: ;3,-4", "TOUT: ;-1,-2,-3,-4"])

        Lnet.PutUniTensors(["Lambda", "Gamma", "Lambda_", "Gamma_conj"], [Lambda, Gamma, Lambda.clone(), Gamma.Conj()])
        Ltemp = Lnet.Launch(True)

        Ltemp_np = tonumpy(Ltemp)
        e = eigs(Ltemp_np.reshape(D**2, D**2).T, k = 1, which = 'LM', return_eigenvectors = False)
        Gamma /= np.real(e)**0.5 # Make the leading eigenvalue == 1

        Lnet.PutUniTensors(["Lambda", "Gamma", "Lambda_", "Gamma_conj"], [Lambda, Gamma, Lambda.clone(), Gamma.Conj()])
        Ltemp = Lnet.Launch(True)

        Rnet.PutUniTensors(["Gamma", "Lambda", "Gamma_conj", "Lambda_"], [Gamma, Lambda, Gamma.Conj(), Lambda.clone()])
        Rtemp = Rnet.Launch(True)

        l = getTMlr(tonumpy(Ltemp).reshape(D**2, D**2).T, D)
        r = getTMlr(tonumpy(Rtemp).reshape(D**2, D**2), D)

        L = cy.UniTensor(cy.from_numpy(scipy.linalg.cholesky(tonumpy(l), check_finite = False)), 0)
        R = cy.UniTensor(cy.from_numpy(scipy.linalg.cholesky(tonumpy(r), check_finite = False).conj().T), 0)

        Linv = cy.UniTensor(cy.linalg.InvM(L.Conj().get_block()), 0)
        Rinv = cy.UniTensor(cy.linalg.InvM(R.get_block()), 0)
        
        net = cy.Network()
        net.FromString(["L_conj: ;-1,1", "Lambda: ;1,2", "R: ;2,-2", "TOUT: ;-1,-2"])
        net.PutUniTensors(["L_conj", "Lambda", "R"], [L.Conj(), Lambda, R])
        temp = net.Launch(True)
        temp.set_rowrank(1)
        s, u, vt = cy.linalg.Svd(temp)
        Lambda = cy.UniTensor(cy.linalg.Diag(s.get_block()) / cy.linalg.Norm(s.get_block()).item(), 0)

        vt, Rinv, Linv, u, Gamma, Lambda = tonumpy(vt),tonumpy(Rinv),tonumpy(Linv),tonumpy(u),tonumpy(Gamma), tonumpy(Lambda)
        Gamma = ncon([vt,Rinv,Gamma,Linv,u],[[-1,1],[1,2],[2,-2,3],[3,4],[4,-3]])
        Gamma /= ncon([Lambda, Gamma, Lambda, Gamma.conj()],[[4,1],[1,2,-1],[4,3],[3,2,-2]])[0,0]**0.5 # make the leading eigenvalue == 1

        return cy.UniTensor(cy.from_numpy(Lambda), 0), cy.UniTensor(cy.from_numpy(Gamma), 0)

    Lambda, Gamma = toLambdaGamma(AL)
    Lambda, Gamma = toCanonical(Lambda, Gamma)

    sqLambda = sqrtm(Lambda)

    Anet = cy.Network()
    Anet.FromString(["sqLambda: ;-1,1", "Gamma: ;1,-2,2", "sqLambda_: ;2,-3", "TOUT: ;-1,-2,-3"])
    Anet.PutUniTensors(["sqLambda", "Gamma", "sqLambda_"], [sqLambda, Gamma, sqLambda.clone()])
    A = Anet.Launch(True)

    return A, Lambda
    
def getMz(A, Lambda):

    '''This is for two sites as a unit cell tensor'''

    A, Lambda = tonumpy(A), tonumpy(Lambda)
    sZ = np.array([[1.0, 0], [0, -1.0]]) # for ising
    sI =  np.array([[1.0, 0], [0, 1.0]])
    meaa, meab = np.kron(sI,sZ), np.kron(sZ,sI)
    norm = ncon([Lambda, Lambda], [[1,2],[1,2]])
    a = ncon([Lambda, A, meaa, A.conj(), Lambda], [[5,6],[5,2,7],[2,4],[6,4,8],[7,8]])/norm
    b = ncon([Lambda, A, meab, A.conj(), Lambda], [[5,6],[5,2,7],[2,4],[6,4,8],[7,8]])/norm
    mz = 0.5*(a+b)
    return mz

# def LeftGaugefixed(A, l, r):

#     '''
#     https://arxiv.org/abs/1810.07006
#     '''

#     def getRank(A, tol = 1e-15):
#         # s = np.linalg.svd(A, full_matrices = True)[1]
#         s = cy.linalg.Svd(A, is_U = False, is_vT = False)[0].get_block()
#         rank = 0
#         for i in range(s.shape()[0]):
#             if s[i].item() > tol:
#                 rank+=1
#         return rank
    
#     def getRightNull(A):
#         # rank = np.linalg.matrix_rank(A)
#         A.set_rowrank(1)
#         rank = getRank(A)

#         # u, s, v = np.linalg.svd(A, full_matrices = True)
#         s, u, v = cy.linalg.Svd(A)
#         # tv = np.transpose(v.conj())
#         tv = v.Conj().permute([1,0])
#         tv.print_diagram()
#         print(tv.shape()[1])
#         print(rank)
#         right_null = tv.get_block()[:, rank:  tv.shape()[1]]
#         return right_null
    
#     def getLeftNull(A):
#         rank = np.linalg.matrix_rank(A)
#         u, s, v = np.linalg.svd(A, full_matrices = True)
#         tu = np.transpose(u.conj())
#         left_null = tu[rank: tu.shape[0], :]
#         return left_null
    
#     D = A.shape()[0]
#     d = A.shape()[1]
#     # temp = ncon([sqrtm(l), A.conj()],[[-2, 1],[1, -3, -1]]).reshape(m, m*d)
#     sqrtl = sqrtm(l); sqrtl.set_labels([-2,1])
#     Aconj = A.Conj(); Aconj.set_labels([1,-3,-1])
#     temp = cy.Contract(sqrtl, Aconj).permute([1, 2, 0]).reshape(D, D*d)
#     #     Vl = linalg.null_space(temp)[:, range(m*(d-1))]
#     Vl = getRightNull(temp)[:, range(m*(d-1))]
#     # Vl = np.linalg.svd(Vl, full_matrices=False)[0].reshape(m,d,m*(d-1))
#     Vl = cy.linalg.Svd(Vl)[0].reshape(m, d, m*(d-1))
#     #     print(ncon([Vl, Vl.conj()], [[1,2,-1], [1,2,-2]])) #Identity
#     #     print(linalg.norm(ncon([sqrtm(l), A.conj(), Vl],[[2, 1], [1, 3, -2], [2,3,-1]]))) # ~= 0
#     return Vl