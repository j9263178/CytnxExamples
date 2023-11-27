
# OBOX
# OBXO
# OXBO
# XOBO

# BOOX
# BOXO
# BXOO
# XBOO

# OOBX
# OOXB
# OXOB
# XOOB
# ============


def braOket(Abra, W, Aket): 
    if len(W.shape) == 4:
        braOket = ncon([Aket, W, Abra.conj()], [[-3,2,-6],[-2,2,-5,1],[-1,1,-4]])
    elif len(W.shape) == 2:
        braOket = ncon([Abra.conj(), W, Aket], [[-1,1,-3], [1,2],[-2,2,-4]])
    return braOket

def pinvL(x, p, Abra, W, Aket, l, r, domain = False):
    AbraOAket = braOket(Abra, W, Aket)
    dimO = len(W.shape)
    D, d, _ = Abra.shape; dw = W.shape[0]

    if np.abs(np.exp(1j*p)-1) < 1e-12 and domain == False:
        if dimO == 4:
            x = x - l*ncon([x,r], [[1,2,3], [3,2,1]])
    def infsum(v):
        if dimO == 4:
            v = v.reshape(D, dw, D)
            v = v - np.exp(1j*p)*ncon([v,AbraOAket],[[1,2,3],[1,2,3,-1,-2,-3]])
        if abs(np.exp(1j*p)-1)<1e-12 and domain == False:
            if dimO == 4:
                v = v + l*ncon([v,r],[[1,2,3], [3,2,1]])
        return v
    if dimO == 4:
        y = bicgstab(LinearOperator((D*dw*D, D*dw*D), matvec=infsum, dtype=np.csingle), b = x.flatten(), tol = 1e-10)[0].reshape(D, dw, D)
    return y

def pinvR(x, p, Abra, W, Aket, l, r, domain = False):
    AbraOAket = braOket(Abra, W, Aket)
    dimO = len(W.shape)
    D, d, _ = Abra.shape; dw = W.shape[0]

    if np.abs(np.exp(1j*p)-1) < 1e-12 and domain == False:
        if dimO == 4:
            x = x-r*ncon([x,l],[[1,2,3], [3,2,1]])
    def infsum(v):
        if dimO == 4:
            v = v.reshape(D, dw, D)
            v = v-np.exp(1j*p)*ncon([v,AbraOAket],[[1,2,3],[-3,-2,-1,3,2,1]])
        if abs(np.exp(1j*p)-1) < 1e-12 and domain == False:
            if dimO == 4:
                v = v+l*ncon([v,r],[[1,2,3], [3,2,1]])
        return v
    if dimO == 4:
        y = bicgstab(LinearOperator((D*dw*D, D*dw*D), matvec=infsum, dtype=np.csingle), b = x.flatten(), tol = 1e-10)[0].reshape(D, dw, D)
    return y

def apply1Deff(B, p, AL, AR, C, O, FL, FR, pinv = "manual"):
    ALOB = braOket(AL, O, B)
    LB = ncon([FL, ALOB], [[1,2], [1,2,-1,-2]])
    AROB = braOket(AR, O, B)
    RB = ncon([AROB, FR], [[-2, -1, 2, 1], [1,2]])
    if pinv == "manual":
        l = ncon([FL,C],[[-1,2],[2,-2]])
        r = ncon([FR,C.conj().T],[[-1,2],[2,-2]]) 
        LB = pinvL(LB,-p,AL,O,AR,l,r)
        l = ncon([C.conj().T,FL],[[-1,1], [1,-2]])
        r = ncon([C,FR], [[-1,1], [1,-2]])
        RB = pinvR(RB,+p,AR,O,AL,l,r)
    By = np.exp(-1j*p)*ncon([LB,AR,O,FR],[[-1,1],[1,2,3],[-2,2],[3,-3]]) + np.exp(1j*p)*ncon([FL,AL,O,RB],[[-1,1],[1,2,3],[-2,2],[3,-3]]) + ncon([FL,B,O,FR],[[-1,1],[1,2,3],[-2,2],[3,-3]])
    return By

def measuretotalspin(B, AL, AR, C, O):
    ALOAL = braOket(AL,O,AL)
    AROAR = braOket(AR,O,AR)
    D, _, _= AL.shape; dw = O.shape[0]
    # FL = np.random.rand(D, D)
    TsfOL = lambda FL : ncon([FL.reshape(D, D),ALOAL], [[1,2],[1,2,-1,-2]])
    FL = eigs(LinearOperator((D*D, D*D), matvec = TsfOL, dtype=np.csingle), k = 1, which = 'LM')[1].reshape(D, D)
    # FR = np.random.rand(D, D)
    TsfOR = lambda FR : ncon([AROAR,FR.reshape(D, D)], [[-2,-1,2,1],[1,2]])
    FR = eigs(LinearOperator((D*D, D*D), matvec = TsfOR, dtype=np.csingle),  k = 1, which = 'LM')[1].reshape(D, D)
    FR /= ncon([FL, C, C.conj(), FR], [[3, 1], [1, 2], [3, 4], [2, 4]])
    By = apply1Deff(B, 0, AL, AR, C, O, FL, FR, pinv = "manual")
    return ncon([B.conj(), By], [[1,2,3],[1,2,3]])