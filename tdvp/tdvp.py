import numpy as np
import cytnx as cy
from ncon import ncon
import scipy
from cmath import sqrt
import scipy.linalg as linalg
from scipy.sparse.linalg import LinearOperator, eigsh, eigs, gmres, bicgstab

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


def tdvpSG(A0, W, dt, numiter, bicg_tol = 1e-12):

    # LAOR = ncon([L,A,W,R],[[2,1,-1],[1,5,4],[2,3,5,-2],[3,4,-3]])
    # lVVl = ncon([l, Vl.conj(), Vl, l], [[-1,1],[1,-2,2],[3,-3,2],[-4,3]])
    # B = ncon([LAOR, lVVl, r],[[1,2,3],[1,2,-2,-1],[3,-3]])

    LAORnet = cy.Network()
    LAORnet.FromString(["L: ;2,1,-1", "A: ;1,5,4", "W: ;2,3,5,-2", "R: ;3,4,-3", "TOUT: ;-1,-2,-3"])
    lVVlnet = cy.Network()
    lVVlnet.FromString(["l: -1;1", "Vl_conj: ;1,-2,2", "Vl: ;3,-3,2", "l_: -4;3","TOUT: ;-1,-2,-3,-4"])
    Bnet = cy.Network()
    Bnet.FromString(["LAOR: ;1,2,3", "lVVl: ;1,2,-2,-1", "r: 3;-3", "TOUT: ;-1,-2,-3"])
    W_ = tonumpy(W) # W_ for getLW and getRW
    D = A0.shape()[0]
    dw = W.shape()[0]
    A = A0.clone()
    mzs = []

    def getLW(A, l, r):

        A, l, r = tonumpy(A), tonumpy(l), tonumpy(r) 

        def getLWaC25(LWa, YLa, AL, L, R):
            m = AL.shape[0]
            def Lcontract(v):
                v = v.reshape(m,m)
                # LWa_TL = ncon([v, TL],[[1,2],[1,-1,2,-2]])
                LWa_TL = ncon([v, AL, AL.conj()],[[1,2],[1,3,-1],[2,3,-2]])
                # LWa_P = ncon([v, R], [[1,2],[1,2]]) * np.eye(m)
                LWa_P = ncon([v, R], [[1,2],[1,2]]) * L
                return v.flatten() - LWa_TL.flatten() + LWa_P.flatten()

            LeftOp = LinearOperator((m**2, m**2), matvec=Lcontract, dtype = np.cdouble)

            # B = YLa - ncon([YLa, R], [[1,2],[1,2]])*np.eye(m)
            B = YLa - ncon([YLa, R], [[1,2],[1,2]]) * L
            LWa_temp, is_conv = bicgstab(LeftOp, B.flatten(), maxiter = 100000, tol=bicg_tol) # x0=LWa.flatten()?
            if is_conv != 0:
                print("bicgstab didn't converge : %d"%is_conv)
                if is_conv<0:
                    print("bicgstab breakdown!")
                    exit(1)
            # LWa_temp = np.real(LWa_temp).reshape(m, m)
            LWa_temp = LWa_temp.reshape(m, m)

            return LWa_temp

        YL = np.zeros([dw, D, D], dtype = np.cdouble)
        # LW_ = LW.copy()
        LW_ = np.zeros([dw, D,D], dtype = np.cdouble)
        # LW_[dw-1] = np.eye(m)
        LW_[dw-1] = l
        R = r
        for a in range(dw-2, -1, -1):
            for b in range(a+1, dw):
                YL[a] += ncon([LW_[b], A, W_[b,a], A.conj()],[[1,2], [1,4,-1], [4,5], [2,5,-2]])
            if W_[a, a, 0, 0] == 0:
                LW_[a] = YL[a]
            elif W_[a, a, 0, 0] == 1:
                LW_[a] = getLWaC25(LW_[a], YL[a], A, l, R)
        
        # return np.asarray(LW_)#, ncon([YL[0], R], [[1,2],[1,2]])
        return cy.UniTensor(cy.from_numpy(np.asarray(LW_)), 0) #, ncon([YL[0], R], [[1,2],[1,2]])

    def getRW(A, l, r):

        A, l, r = tonumpy(A), tonumpy(l), tonumpy(r) 
        
        def getRWaC25(RWa, YRa, AR, L, R):
            m = AR.shape[0]
            def Rcontract(v):
                v = v.reshape(m,m)
                # RWa_TR = ncon([TR, v],[[-1,1,-2,2], [1,2]])
                RWa_TR = ncon([v, AR, AR.conj()],[[1,2], [-1,3,1], [-2,3,2]])
                # RWa_P = ncon([L, v], [[1,2],[1,2]])*np.eye(m)
                RWa_P = ncon([L, v], [[1,2],[1,2]]) * R
                return v.flatten() - RWa_TR.flatten() + RWa_P.flatten()

            RightOp = LinearOperator((m**2, m**2), matvec=Rcontract, dtype = np.cdouble)

            # B = YRa - ncon([L, YRa], [[1,2],[1,2]]) * np.eye(m)
            B = YRa - ncon([L, YRa], [[1,2],[1,2]]) * R
            RWa_temp, is_conv = bicgstab(RightOp, B.flatten(), maxiter = 100000, tol=bicg_tol) # x0=RWa.flatten()?
            if is_conv != 0:
                print("bicgstab didn't converge : %d"%is_conv)
                if is_conv<0:
                    print("bicgstab breakdown!")
                    exit(1)
            # RWa_temp = np.real(RWa_temp).reshape(m, m)
            RWa_temp = RWa_temp.reshape(m, m)

            return RWa_temp

        YR = np.zeros([dw, D, D], dtype = np.cdouble)
        # RW_ = RW.copy()
        RW_ = np.zeros([dw, D, D], dtype = np.cdouble)
        # RW_[0] = np.eye(m)
        RW_[0] = r
        L = l
        for a in range(1, dw):
            for b in range(a-1, -1, -1):
                YR[a] += ncon([RW_[b], A, W_[a,b], A.conj()],[[1,2], [-1,4,1], [4,5], [-2,5,2]])
            if W_[a, a, 0, 0] == 0:
                RW_[a] = YR[a]
            elif W_[a, a, 0, 0] == 1:
                RW_[a] = getRWaC25(RW_[a], YR[a], A, L, r)

        # return np.asarray(RW_) #, ncon([L, YR[-1]], [[1,2],[1,2]])
        return cy.UniTensor(cy.from_numpy(np.asarray(RW_)), 0)

    def getB(A):
        l, r = getlr(A)
        Vl = LeftGaugefixed(A, l, r)
        L, R = getLW(A, l, r), getRW(A, l, r)
        LAORnet.PutUniTensors(["L","A","W","R"], [L, A, W, R], False)
        LAOR = LAORnet.Launch(True)
        l, r = getInverselr(l, r)
        lVVlnet.PutUniTensors(["l", "Vl_conj", "Vl", "l_"], [l, Vl.Conj(), Vl, l.clone()], False)
        lVVl = lVVlnet.Launch(True)
        Bnet.PutUniTensors(["LAOR", "lVVl", "r"], [LAOR, lVVl, r], False)
        B = Bnet.Launch(True)
        return B

    '''
    Change uMPS (usaully AL or AR from VUMPS algorithm) to symmetric gauge to balance the condition number of left and right eigenvector.
    '''

    for ite in range(numiter):

        A, Lambda = toSymmetricGauge(A)
        print("Iteration : %d"%ite)
        mzs.append(getMz(A, Lambda))
        B1 = getB(A); A1 = A-1j*(1/2)*B1*dt
        B2 = getB(A1); A2 = A-1j*(1/2)*B2*dt
        B3 = getB(A2); A3 = A-1j*B3*dt
        B4 = getB(A3)
        A = A - 1j * (1/6) * (B1 + 2*B2 + 2*B3 + B4) * dt

    return np.asarray(mzs)

if __name__ == '__main__':
    
    from vumpsMPO import vumpsMPO

    def ising(h = 10):
        d = 2
        dw = 3
        sx = cy.physics.pauli('x').real()
        sz = cy.physics.pauli('z').real()
        sI = cy.eye(d)
        M = cy.zeros([dw, dw, d, d])
        M[0,0] = sI; M[2,2] = sI
        M[0,1] = sx; M[1,2] = -sx
        M[0,2] = h * sz
        M = cy.UniTensor(M, 0)
        M.permute_([1,0,2,3]) # In the paper the MPO is of lower tridiagonal
        M_ = M.clone()
        M.set_labels([-1,1,-3,-5]); M_.set_labels([1,-2,-4,-6])
        MM = cy.Contract(M, M_)
        MM.permute_([0,3,1,4,2,5])
        MM.reshape_(dw, dw, 4, 4)
        # MM.print_diagram()
        return MM

    ''' Quench dynamics in TFIM (h = 10 ~ h = 3) '''

    d = 4
    D = 10
    M = ising(h = 10)

    C = cy.UniTensor(cy.linalg.Diag(cy.random.normal([D], 0., 1.)), 1)
    C =  C / C.get_block_().Norm().item()
    AL = (cy.linalg.Svd(cy.UniTensor(cy.random.normal([D*d, D],0.,1.), 1))[1]).reshape(D, d, D)
    AR = (cy.linalg.Svd(cy.UniTensor(cy.random.normal([D, D*d],0.,1.), 1))[2]).reshape(D, d, D)

    AL, C, AR, LW, RW, Energy = vumpsMPO(M, AL, AR, C, maxit = 50) # Find ground state with VUMPS

    M = ising(h = 3)
    datas = tdvpSG(A0 = AL, W = M, dt = 0.01, numiter = 400, bicg_tol = 1e-12)
    savename = "tdvp_ising_D%d"%D
    np.save(savename, datas)

    