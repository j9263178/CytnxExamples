## Follow SciPost Phys. Lect. Notes 7 (2019)

# const MPSOperator = Union{Matrix, Dims4Array} ## Dims 2 or 4
# const MPSEnvironment = Union{Matrix,Dims3Array} ## Dims 2 or 3

import numpy as np
from ncon import ncon
from scipy.linalg import null_space
from scipy.sparse.linalg import LinearOperator, bicgstab, eigs

"""
    braOket(Abra, W, Aket)
Compute the channel operator TW, where W is an Operator.
"""

def braOket(Abra, W, Aket):
    
    if len(W.shape) == 4:
        braOket = ncon([Aket, W, Abra.conj()], [[-3,2,-6],[-2,2,-5,1],[-1,1,-4]])
    elif len(W.shape) == 2:
        braOket = ncon([Abra.conj(), W, Aket], [[-1,1,-3], [1,2],[-2,2,-4]])
    return braOket

# function braOket(Abra::Dims3Array, W::MPSOperator, Aket::Dims3Array)
#     if ndims(W) == 4
#         braOket =  ncon([Aket,W,conj(Abra)],[[-3,2,-6],[-2,2,-5,1],[-1,1,-4]])
#     elseif  ndims(W) ==  2
#         braOket =  ncon([conj(Abra), W, (Aket)], [[-1,1,-3], [1,2],[-2,2,-4]])
#     end
#     return braOket
# end

#################### pinvL, pinvR
"""
    pinvL(x,p,Abra,W,Aket,l,r; domain)
"""

def pinvL(x, p, Abra, W, Aket, l, r, domain = False):
    AbraOAket = braOket(Abra, W, Aket)
    dimO = len(W.shape)
    D, d, _ = Abra.shape; dw = W.shape[0]

    if np.abs(np.exp(1j*p)-1) < 1e-12 and domain == False:
        if dimO == 4:
            x = x - l*ncon([x,r], [[1,2,3], [3,2,1]])
        elif dimO == 2:
            x = x - l*ncon([x,r],[[1,2], [2,1]])
    def infsum(v):

        if dimO == 4:
            v = v.reshape(D, dw, D)
            v = v - np.exp(1j*p)*ncon([v,AbraOAket],[[1,2,3],[1,2,3,-1,-2,-3]])
        elif dimO == 2:
            v = v.reshape(D, D)
            v = v - np.exp(1j*p)*ncon([v,AbraOAket],[[1,2],[1,2,-1,-2]])
        if abs(np.exp(1j*p)-1)<1e-12 and domain == False:
            if dimO == 4:
                v = v + l*ncon([v,r],[[1,2,3], [3,2,1]])
            elif dimO == 2:
                v = v + l*ncon([v,r],[[1,2], [2,1]])
        return v
    if dimO == 4:
        y = bicgstab(LinearOperator((D*dw*D, D*dw*D), matvec=infsum, dtype=np.csingle), b = x.flatten(), tol = 1e-10)[0].reshape(D, dw, D)
    elif dimO == 2:
        y = bicgstab(LinearOperator((D*D, D*D), matvec=infsum, dtype=np.csingle), b = x.flatten(), tol = 1e-10)[0].reshape(D, D)
    return y


# function pinvL(x::MPSEnvironment,p::Real,Abra::Dims3Array, W::MPSOperator,
#      Aket::Dims3Array,l::Union{MPSEnvironment, Nothing},r::Union{MPSEnvironment, Nothing}; domain = false)
#     AbraOAket = braOket(Abra, W, Aket)
#     dimO = ndims(W)
#     if abs(exp(1im*p)-1)<1e-12 && domain == false
#         if dimO == 4
#             x = x-l*scalar(ncon([x,r],[[1,2,3], [3,2,1]]))
#         elseif dimO == 2
#             x = x-l*scalar(ncon([x,r],[[1,2], [2,1]]))
#         end
#     end
#     y, info= linsolve(x;tol = 1e-10, maxiter = 400) do x
#         if dimO == 4
#             y=x-exp(1im*p)*ncon([x,AbraOAket],[[1,2,3],[1,2,3,-1,-2,-3]])
#         elseif dimO == 2
#             y=x-exp(1im*p)*ncon([x,AbraOAket],[[1,2],[1,2,-1,-2]])
#         end
#         if abs(exp(1im*p)-1)<1e-12 && domain == false
#             if dimO == 4
#                 y= y+l*scalar(ncon([x,r],[[1,2,3], [3,2,1]]))
#             elseif dimO == 3
#                 y= y+l*scalar(ncon([x,r],[[1,2], [2,1]]))
#             end
#         end
#         return y
#     end
#     #@assert info.converged != 0
#     info.converged == 0 ? @warn("pinvL not converge!!!") : nothing
#     return y
# end



"""
    pinvR(x,p,Abra,W,Aket,l,r; domain)
"""

def pinvR(x, p, Abra, W, Aket, l, r, domain = False):
    AbraOAket = braOket(Abra, W, Aket)
    dimO = len(W.shape)
    D, d, _ = Abra.shape; dw = W.shape[0]

    if np.abs(np.exp(1j*p)-1) < 1e-12 and domain == False:
        if dimO == 4:
            x = x-r*ncon([x,l],[[1,2,3], [3,2,1]])
        elif dimO == 2:
            x = x-r*ncon([x,l],[[1,2], [2,1]])
    def infsum(v):
        if dimO == 4:
            v = v.reshape(D, dw, D)
            v = v-np.exp(1j*p)*ncon([v,AbraOAket],[[1,2,3],[-3,-2,-1,3,2,1]])
        elif dimO == 2:
            v = v.reshape(D, D)
            v = v-np.exp(1j*p)*ncon([v,AbraOAket],[[1,2],[-2,-1,2,1]])
        if abs(np.exp(1j*p)-1) < 1e-12 and domain == False:
            if dimO == 4:
                v = v+l*ncon([v,r],[[1,2,3], [3,2,1]])
            elif dimO == 2:
                v = v+l*ncon([v,r],[[1,2], [2,1]])
        return v
    if dimO == 4:
        y = bicgstab(LinearOperator((D*dw*D, D*dw*D), matvec=infsum, dtype=np.csingle), b = x.flatten(), tol = 1e-10)[0].reshape(D, dw, D)
    elif dimO == 2:
        y = bicgstab(LinearOperator((D*D, D*D), matvec=infsum, dtype=np.csingle), b = x.flatten(), tol = 1e-10)[0].reshape(D, D)
    return y
# function pinvR(x::MPSEnvironment,p,Abra::Dims3Array, O::MPSOperator, 
#     Aket::Dims3Array,l::Union{MPSEnvironment, Nothing},r::Union{MPSEnvironment, Nothing}; domain = false)
#     dimO = ndims(O)
#     AbraOAket = braOket(Abra, O, Aket)
#     if abs(exp(1im*p)-1) < 1e-12 && domain == false
#         if dimO == 4
#             x = x-r*scalar(ncon([x,l],[[1,2,3], [3,2,1]]))
#         elseif dimO == 2
#             x = x-r*scalar(ncon([x,l],[[1,2], [2,1]]))
#         end

#     end
#     y, info= linsolve(x;tol = 1e-10, maxiter = 400) do x
#         if dimO == 4
#             y=x-exp(1im*p)*ncon([x,AbraOAket],[[1,2,3],[-3,-2,-1,3,2,1]])
#         elseif dimO == 2
#             y=x-exp(1im*p)*ncon([x,AbraOAket],[[1,2],[-2,-1,2,1]])
#         end
#         if abs(exp(1im*p)-1)<1e-12 && domain == false
#             if dimO == 4
#                 y= y+l*scalar(ncon([x,r],[[1,2,3], [3,2,1]]))
#             elseif dimO == 2
#                 y= y+l*scalar(ncon([x,r],[[1,2], [2,1]]))
#             end
#         end
#         return y
#     end
#     info.converged == 0 ? @warn("pinvR not converge!!!") : nothing
#     return y
# end


# function pinvLJulia(x::MPSEnvironment,p, Abra::Dims3Array, O::MPSEnvironment, Aket::Dims3Array;)
#     dimO = ndims(O)
#     if dimO == 4
#         D,dw = size(x)
#         mat_braOket = reshape(braOket(Abra, O, Aket), (D^2*dw,D^2*dw))
#         eye = Matrix(I,size(mat_braOket))
#         mat_inv = pinv(eye - exp(1im*p)*transpose(mat_braOket))
#         invL = reshape(mat_inv, (D,dw,D,D,dw,D))
#         y = ncon([invL, x], [[-1,-2,-3,1,2,3], [1,2,3]])
#     elseif dimO == 2
#         D, = size(x)
#         mat_braOket = reshape(braOket(Abra,O,Aket), D^2,D^2)
#         eye = Matrix(I,size(mat_braOket))
#         mat_inv = pinv(eye - exp(1im*p)*transpose(mat_braOket))
#         invL = reshape(mat_inv, (D,D,D,D))
#         y = ncon([invL, x], [[-1,-2,1,2], [1,2]])
#     end
# end

# function pinvRJulia(x::MPSEnvironment,p, Abra::Dims3Array, O::MPSOperator, Aket::Dims3Array)
#     dimO = ndims(O)
#     if dimO == 4
#         D,dw = size(x)
#         mat_braOket = reshape(braOket(Abra, O, Aket), (D^2*dw,D^2*dw))
#         eye = Matrix(I,size(mat_braOket))
#         mat_inv = pinv(eye - exp(1im*p)*mat_braOket)
#         invL = reshape(mat_inv, (D,dw,D,D,dw,D))
#         y = ncon([invL, x], [[-3,-2,-1,3,2,1], [1,2,3]])
#     elseif dimO == 2
#         D, = size(x)
#         mat_braOket = reshape(braOket(Abra,O,Aket), (D^2,D^2))
#         eye = Matrix(I,size(mat_braOket))
#         mat_inv = pinv(eye - exp(1im*p)*mat_braOket)
#         invL = reshape(mat_inv, (D,D,D,D))
#         y = ncon([invL, x], [[-2,-1,2,1], [1,2]])
#     end
# end


#################### apply_Heff, apply_1Deff, measure_charge
"""
    applyHeff(B,p,AL,AR,C,O,FL,FR)
MPO version of applyHeff
"""
def applyHeff(B, p, AL, AR, C, O, FL, FR, pinv = "manual"):
    ALOB = braOket(AL, O, B)
    LB = ncon([FL,ALOB], [[1,2,3], [1,2,3,-1,-2,-3]])
    AROB = braOket(AR, O, B)
    RB = ncon([FR,AROB],[[1,2,3], [-3,-2,-1,3,2,1]])
    if pinv == "manual":
        l = ncon([FL,C],[[-1,-2,3],[3,-3]])
        r = ncon([FR,C.conj().T],[[-1,-2,3],[3,-3]])
        LB = pinvL(LB, -p, AL, O, AR, l, r)

        l = ncon([C.conj().T,FL],[[-1,1], [1,-2,-3]])
        r = ncon([C,FR], [[-1,1], [1,-2,-3]]) 
        RB = pinvR(RB, +p, AR, O, AL, l, r)
        By = np.exp(-1j*p)*ncon([LB,AR,O,FR],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]])+np.exp(1j*p)*ncon([FL,AL,O,RB],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]])+ncon([FL,B,O,FR],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]])
    return By
# function applyHeff(B::Dims3Array,p,AL::Dims3Array,AR::Dims3Array,C::AbstractMatrix,
#     O::MPSOperator,FL::MPSEnvironment,FR::MPSEnvironment; pinv = "manual")
#     ALOB = braOket(AL,O,B)
#     LB = ncon([FL,ALOB], [[1,2,3], [1,2,3,-1,-2,-3]])
#     AROB = braOket(AR,O,B)
#     RB = ncon([FR,AROB],[[1,2,3], [-3,-2,-1,3,2,1]])

#     if pinv == "manual"
#         l = ncon([FL,C],[[-1,-2,3],[3,-3]])
#         r = ncon([FR,C'],[[-1,-2,3],[3,-3]])
#         LB = pinvL(LB,-p,AL,O,AR,l,r)

#         l = ncon([C',FL],[[-1,1], [1,-2,-3]])
#         r = ncon([C,FR], [[-1,1], [1,-2,-3]]) 
#         RB=pinvR(RB,+p,AR,O,AL,l,r);
#     elseif pinv == "julia"
#         @warn("Uisng Julia's inverse, and the computational cost is much higher than linsolve!")
#         LB = pinvLJulia(LB,-p,AL,O,AR)
#         RB = pinvRJulia(RB,+p,AR,O,AL)
#     end

#     By= exp(-1im*p)*ncon([LB,AR,O,FR],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]]) +
#         exp(1im*p)*ncon([FL,AL,O,RB],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]]) +
#         ncon([FL,B,O,FR],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]])
    
#     return By   
# end

"""
    applydomainHeff(B,p,AL1,AR2,C,O,FL,FR)
"""
def applydomainHeff(B, p, AL1, AR2, C, O, FL, FR, pinv = "manual"):
    AL1OB = braOket(AL1, O, B)
    AL1OB = braOket(AL1,O,B)
    LB = ncon([FL,AL1OB], [[1,2,3], [1,2,3,-1,-2,-3]])
    AR2OB = braOket(AR2,O,B)
    RB = ncon([FR,AR2OB],[[1,2,3], [-3,-2,-1,3,2,1]])
    if pinv == "manual":
        l = r = None
        LB = pinvL(LB, -p, AL1, O, AR2, l, r, domain = True)
        RB = pinvR(RB, +p, AR2, O, AL1, l, r, domain = True)
    By = np.exp(-1j*p)*ncon([LB,AR2,O,FR],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]])+np.exp(1j*p)*ncon([FL,AL1,O,RB],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]])+ncon([FL,B,O,FR],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]])
    
    return By   
# function applydomainHeff(B::Dims3Array,p,AL1::Dims3Array,AR2::Dims3Array,C::AbstractMatrix,
#     O::Dims4Array,FL::Dims3Array,FR::Dims3Array; pinv = "manual")
#     AL1OB = braOket(AL1,O,B)
#     LB = ncon([FL,AL1OB], [[1,2,3], [1,2,3,-1,-2,-3]])
#     AR2OB = braOket(AR2,O,B)
#     RB = ncon([FR,AR2OB],[[1,2,3], [-3,-2,-1,3,2,1]])
#     if pinv == "manual"
#         l = r = nothing
#         LB = pinvL(LB,-p,AL1,O,AR2,l,r; domain = true);
#         RB=pinvR(RB,+p,AR2,O,AL1,l,r; domain = true);
#     elseif pinv == "julia"
#         @warn("Uisng Julia's inverse, and the computational cost is much higher than linsolve!")
#         LB = pinvLJulia(LB,-p,AL1,O,AR2)
#         RB = pinvRJulia(RB,+p,AR2,O,AL1)
#     end

#     By= exp(-1im*p)*ncon([LB,AR2,O,FR],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]]) +
#         exp(1im*p)*ncon([FL,AL1,O,RB],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]]) +
#         ncon([FL,B,O,FR],[[-1,1,2],[2,5,4],[1,5,3,-2],[4,3,-3]])
    
#     return By   
# end

"""
    apply1Deff(B,AL,AR,C,O,FL,FR;)
"""
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

# function apply1Deff(B::Dims3Array,p,AL::Dims3Array,AR::Dims3Array,C::AbstractMatrix,O::AbstractMatrix,
#     FL,FR; pinv = "manual")
#     ALOB = braOket(AL,O,B)
#     LB = ncon([FL,ALOB], [[1,2],[1,2,-1,-2]])
#     AROB = braOket(AR,O,B)
#     RB = ncon([AROB, FR],[[-2,-1,2,1],[1,2]])
#     if pinv == "manual"
#         l = ncon([FL,C],[[-1,2],[2,-2]])
#         r = ncon([FR,C'],[[-1,2],[2,-2]])
#         LB = pinvL(LB,-p,AL,O,AR,l,r)
#         l = ncon([C',FL],[[-1,1], [1,-2]])
#         r = ncon([C,FR], [[-1,1], [1,-2]]) 
#         RB=pinvR(RB,+p,AR,O,AL,l,r);
#     elseif pinv == "julia"
#         @warn("Uisng Julia's inverse, and the computational cost is much higher than linsolve!")
#         LB = pinvLJulia(LB,-p,AL,O,AR)
#         RB = pinvRJulia(RB,+p,AR,O,AL)
#     end

#     By= exp(-1im*p)*ncon([LB,AR,O,FR],[[-1,1],[1,2,3],[-2,2],[3,-3]]) +
#         exp(1im*p)*ncon([FL,AL,O,RB],[[-1,1],[1,2,3],[-2,2],[3,-3]]) +
#         ncon([FL,B,O,FR],[[-1,1],[1,2,3],[-2,2],[3,-3]])
#     return By   
# end


"""
    measurecharge(B,AL,AR,C,O)
"""

def measurecharge(B, AL, AR, C, O):
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

# function measurecharge(B::Dims3Array,AL::Dims3Array,AR::Dims3Array,C::AbstractMatrix,O::AbstractMatrix)
#     ALOAL = braOket(AL,O,AL);
#     AROAR = braOket(AR,O,AR);
#     D, = size(AL)
#     FL = randn(eltype(AL), D,D);
#     λs, FLs, info = eigsolve(FL, 1, :LM; ishermitian = false) do FL
#         FL = ncon([FL,ALOAL], [[1,2],[1,2,-1,-2]])
#     end
#     λL = λs[1]; FL = FLs[1];
#     FR = randn(eltype(AL), D,D);
#     λs, FRs, info = eigsolve(FR, 1, :LM; ishermitian = false) do FR
#         FR = ncon([AROAR,FR], [[-2,-1,2,1],[1,2]])
#     end
#     λR = λs[1]; FR = FRs[1];
#     FR ./= @tensor scalar(FL[c,a]*C[a,a']*conj(C[c,c'])*FR[a',c']);

#     By = apply1Deff(B,0,AL,AR,C,O,FL,FR; pinv = "manual");
#     return (scalar(ncon([conj(B),By], [[1,2,3],[1,2,3]])))
# end

"""
    getnullspace(AL)
Given `AL`, return the its nullspace `VL` and its effective dimension `nL` 
"""

def getnullspace(AL):
    D, d, _ = AL.shape
    L = np.transpose(AL.conj(), (2,0,1)).reshape(D, D*d)
    VL = null_space(L)
    nL = VL.shape[1]
    VL = VL.reshape(D, d, nL)
    return VL, nL

# function getnullspace(AL::Dims3Array)
#     D_mps, d_mps, = size(AL)
#     L = reshape(permutedims(conj(AL), (3,1,2)), (D_mps,D_mps*d_mps))
#     VL=nullspace(L); nL= size(VL)[2]; VL=reshape(VL,(D_mps, d_mps, nL));
#     return VL, nL
# end



"""
    excitation(W,AL,AR,C,FL,FR,num_ω,p;charge,Cstring,domain,Fstring,verbose)
Return a Dict `data` with data["p"] = p; data["ϕ"] = ϕ; data["ω"] = abs_ωs
"""

def excitation(W, AL, AR, C, FL, FR, num_w, p, charge = False, Cstring = None, domain = None, Fstring = None, verbose = False):
    D = AL.shape[0]
    VL, nL = getnullspace(AL)
    if domain == True:
        applyH = applydomainHeff
        AR2 = ncon([Fstring[0],AR],[[-2,2],[-1,2,-3]]); AR2OAR2 = braOket(AR2,W,AR2)
        FR2 = ncon([Fstring[0],FR],[[-2,2],[-1,2,-3]])
    else:
        applyH = applyHeff
        AR2 = AR; FR2 = FR

    def operator(X):
        B = ncon([VL, X.reshape(nL, D)],[[-1,-2,1],[1,-3]])
        By = applyH(B, p*np.pi, AL, AR2, C, W, FL, FR2, pinv = "manual")   
        Heff_X = ncon([By, VL.conj()],
                    [[1,2,-2],[1,2,-1]])
        return Heff_X
    ws, excits = eigs(LinearOperator((D*nL, D*nL), matvec=operator, dtype=np.csingle), v0 = np.random.rand(nL, D), k = num_w)
    phi = np.angle(ws)/np.pi
    abs_ws = np.abs(ws)
    data = {}
    if charge == True:
        if len(Cstring) == 1:
            Ceq = []; Cdiff = []
            ZZ = Cstring[0]
            for i in range(num_w):
                excit = (excits.T)[i].reshape(nL, D)
                B = ncon([VL,excit],[[-1,-2,1],[1,-3]])
                tmp = np.real(measurecharge(B,AL,AR,C,ZZ))
                print("charge = ",(tmp), " ") if verbose == True else None
                Ceq.append(i) if (tmp > 0) else Cdiff.append(i)
            data["p"] = p; data["phi_Ceq"] = phi[Ceq]; data["w_Ceq"] = abs_ws[Ceq]
            data["p"] = p; data["phi_Cdiff"] = phi[Cdiff]; data["w_Cdiff"] = abs_ws[Cdiff]
        elif len(Cstring) == 2:
            Cee = []; Ceo = []; Coo = []
            ZZ = Cstring[0]; IZ = Cstring[1]
            for i in range(num_w):
                excit = (excits.T)[i].reshape(nL, D)
                B = ncon([VL,excit],[[-1,-2,1],[1,-3]])
                CZZ = np.real(measurecharge(B,AL,AR,C,ZZ))
                print("CZZ = ",(CZZ), " ") if verbose == True else None
                CIZ = np.real(measurecharge(B,AL,AR,C,IZ))
                print("CIZ = ",(CIZ), " ") if verbose == True else None
                if CZZ > 0:
                    Cee.append(i) if (CIZ > 0) else Coo.append(i)
                else:
                    Ceo.append(i)
                    
            data["p"] = p; data["phi_Cee"] = phi[Cee]; data["w_Cee"] = abs_ws[Cee]
            data["p"] = p; data["phi_Ceo"] = phi[Ceo]; data["w_Ceo"] = abs_ws[Ceo]
            data["p"] = p; data["phi_Coo"] = phi[Ceo]; data["w_Coo"] = abs_ws[Coo]
    else:
        data["p"] = p; data["phi"] = phi[:num_w]; data["w"] = abs_ws[:num_w]
    print("p = %.6e,  ω =  %.6e, ϕ = %.6e"%(p, abs_ws[0], phi[0])) if verbose == True else None
    return data