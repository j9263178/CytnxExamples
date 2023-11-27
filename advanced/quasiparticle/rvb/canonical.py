import numpy as np
from scipy.linalg import qr, svd, norm, polar
from scipy.sparse.linalg import eigs
from ncon import ncon

"""
    safesign(x::Number)
will be used to make QR decomposition unique
"""

def safesign(x):
    newx = np.zeros(len(x))
    for i in range(len(x)):
        newx[i] = (1 if np.abs(x[i])<=1e-10 else np.sign(x[i]))
    return newx

# safesign(x::Number) = iszero(x) ? one(x) : sign(x) # will be used to make QR decomposition unique

##### modifed version of QR and LQ decomposition

"""
    qrpos(A)
Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""

def qrpos(A):
    Q, R = qr(A, mode = 'economic')
    phases = np.diag(safesign(np.diag(R)))
    Q = Q@phases
    R = phases.conj()@R
    return Q, R

# def qrpos(A):
#     Q, R = cy.linalg.QR(A)
#     phases = safesign(cy.linalg.Diag(R))
#     Q = cy.linalg.Matmul(Q, phase)
#     R = cy.linalg.Matmul(phase.Conj(), R)
#     return Q, R

# qrpos(A) = qrpos!(copy(A))
# function qrpos!(A)
#     F = qr!(A)
#     Q = Matrix(F.Q)
#     R = F.R
#     phases = safesign.(diag(R))
#     rmul!(Q, Diagonal(phases))
#     lmul!(Diagonal(conj!(phases)), R)
#     return Q, R
# end

"""
    lqpos(A)
Returns a LQ decomposition, i.e. a lower triangular `L` and isometric `Q` matrix, where `L`
is guaranteed to have positive diagonal elements.
"""


def lqpos(A):
    Q, R = qr(A.T, mode = 'economic')
    Q = Q.T
    L = R.T
    phases = np.diag(safesign(np.diag(L)))
    Q = phases@Q
    L = L@phases.conj()
    return L, Q

# def lqpos(A):
#     Q, R = cy.linalg.QR(A.transpose())
#     Q = Q.transpose()
#     L = R.transpose()
#     phases = safesign(cy.linalg.Diag(L))
#     Q = cy.linalg.Matmul(phases, Q)
#     L = cy.linalg.Matmul(L, phases.Conj())
#     return L, Q

# lqpos(A::AbstractMatrix) = lqpos!(copy(A))
# function lqpos!(A)
#     F = qr!(Matrix(transpose(A)))
#     Q = transpose(Matrix(F.Q))
#     L = transpose(Matrix(F.R))
#     phases = safesign.(diag(L))
#     lmul!(Diagonal(phases), Q)
#     rmul!(L, Diagonal(conj!(phases)))
#     return L, Q
# end

########### Mixed Gauge: leftorth, rightorth, mixcanonical, and min_AC_C
"""
    leftorth(A, [C]; kwargs...)
Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `C` and
a scalar factor `λ` such that ``λ AL^s C = C A^s``, where an initial guess for `C` can be
provided.
"""

def leftorth(A, C = None, tol = 1e-12, maxiter = 100):
    D, d, _ = A.shape
    if C is None:
        C = np.eye(D)
    ## find better initial guess G
    TsfA = ncon([A, A.conj()],[[-1,1,-3],[-2,1,-4]]).reshape(D**2, D**2)
    # e, rho = eigs(TsfA, v0 = G.conj().T@G, tol = tol, maxiter = 1) ## Find the fixed point of ∑ᵢAⁱ⊗Aⁱ 
    e, rho = eigs(TsfA, k=1, tol = tol, maxiter = 1) ## Find the fixed point of ∑ᵢAⁱ⊗Aⁱ 
    rho = rho.reshape(D, D)
    rho = rho+rho.conj().T ## rho = C'*C is hermitian
    rho /= np.trace(rho) ## enforce tr(ρ) = 1

    # If ρ is not exactly positive definite, cholesky will fail
    u, s, vt = svd(rho)
    C = np.diag(np.sqrt(s))@vt ## given ρ = C'*C, find C

    _, C = qrpos(C) # I don't know why

    ## Algorithm 1
    Q, R = qrpos((C@A.reshape(D, d*D)).reshape(D*d, D))
    AL = Q.reshape(D, d, D)
    R /= norm(R)
    numiter = 1
    while norm(C-R) > tol and numiter < maxiter:
        # G = R
        mixTsf = ncon([A, A.conj()],[[-1,1,-3],[-2,1,-4]]).reshape(D**2, D**2)
        C = eigs(mixTsf, v0 = R, k = 1, which = 'LM', tol = tol, maxiter = 1)[1].reshape(D, D)
        _, C = qrpos(C)
        # The previous three lines can speed up the process when C is still very far from the correct
        # gauge transform, refer to equation (30) in the paper.
        Q, R = qrpos((C@A.reshape(D, d*D)).reshape(D*d, D))
        AL = Q.reshape(D, d, D)
        lambd = norm(R)
        R /= lambd
        numiter += 1
    C = R
    return AL, C, lambd

# function leftorth(A::Dims3Array, C = Matrix{eltype(A)}(I, size(A,1), size(A,1)); tol = 1e-12, maxiter = 100, kwargs...)
#     # find better initial guess C
#     λ2s, ρs, info = eigsolve(C'*C, 1, :LM; ishermitian = false, tol = tol, maxiter = 1, kwargs...) do ρ
#         @tensor ρE[a,b] := ρ[a',b']*A[b',s,b]*conj(A[a',s,a]) ## Find the fixed pt of ∑ᵢAⁱ⊗Aⁱ      
#         return ρE
#     end
#     ρ = ρs[1] + ρs[1]' ## ρ = C'*C is hermitian
#     ρ ./= tr(ρ) ## enforce tr(ρ) = 1
#     # C = cholesky!(ρ).U
#     # If ρ is not exactly positive definite, cholesky will fail
#     F = svd!(ρ)
#     C = lmul!(Diagonal(sqrt.(F.S)), F.Vt) ## given ρ = C'*C, find C
#     _, C = qrpos!(C) # I don't know why

#     ## Algorithm1 
#     D, d, = size(A)
#     Q, R = qrpos!(reshape(C*reshape(A, D, d*D), D*d, D))
#     AL = reshape(Q, D, d, D)
#     λ = norm(R)
#     rmul!(R, 1/λ)
#     numiter = 1
#     while norm(C-R) > tol && numiter < maxiter
#         # C = R
#         λs, Cs, info = eigsolve(R, 1, :LM; ishermitian = false, tol = tol, maxiter = 1, kwargs...) do X
#             @tensor Y[a,b] := X[a',b']*A[b',s,b]*conj(AL[a',s,a])
#             return Y
#         end
#         _, C = qrpos!(Cs[1])
#         # The previous lines can speed up the process when C is still very far from the correct
#         # gauge transform, it finds an improved value of C by finding the fixed point of a
#         # 'mixed' transfer matrix composed of `A` and `AL`, even though `AL` is also still not
#         # entirely correct. Therefore, we restrict the number of iterations to be 1 and don't
#         # check for convergence
#         Q, R = qrpos!(reshape(C*reshape(A, D, d*D), D*d, D)) ## do QR decomposition iteratively
#         AL = reshape(Q, D, d, D)
#         λ = norm(R)
#         rmul!(R, 1/λ)
#         numiter += 1
#     end
#     C = R
#     return AL, C, λ
# end


"""
    rightorth(A, [G]; kwargs...)
Given an MPS tensor `A`, return a gauge transform G, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ G AR^s = A^s G``, where an initial guess for `G` can be
provided.
"""

def rightorth(A, tol = 1e-12, maxiter = 100):
    ## simply permute A and C for leftorth!
    AL, C, lambd = leftorth(np.transpose(A,(2,1,0)), tol = tol)
    return np.transpose(AL,(2,1,0)), np.transpose(C,(1,0)), lambd

# function rightorth(A::Dims3Array, C = Matrix{eltype(A)}(I, size(A,1), size(A,1)); tol = 1e-12, kwargs...)
#     ## simply permute A and C for leftorth!
#     AL, C, λ = leftorth(permutedims(A,(3,2,1)), permutedims(C,(2,1)); tol = tol, kwargs...)
#     return permutedims(AL,(3,2,1)), permutedims(C,(2,1)), λ
# end


"""
    mixcanonical(A)
Transform a mps `A` into the mixe canonical form and return `AL, AR, C` 
"""
def mixcanonical(A):
    ## see Algorithm2
    AL, _, _ = leftorth(A)
    AR, C, _ = rightorth(AL)
    return AL, AR, C

# function mixcanonical(A::Dims3Array)
#     ## see Algorithm2
#     AL, = leftorth(A)
#     AR, C, = rightorth(AL)
#     return AL, AR, C
# end

"""
    min_AC_C(AC, C)
Given `AC, C` and then return `AL,AR` along with the error `errL, errR`
"""

def min_AC_C(AC, C):
    ## algorithm 5
    D, d, _ = AC.shape
    QAC, RAC = qrpos(AC.reshape(D*d, D)) ## polar left for AC
    QC, RC = qrpos(C)
    AL = (QAC@QC.conj().T).reshape(D, d, D)
    errL = norm(RAC-RC) ## not sure why
    LAC, QAC = lqpos(AC.reshape(D, d*D)) ## polar left for AC
    LC, QC = lqpos(C)
    AR = (QC.conj().T@QAC).reshape(D, d, D)
    errR = norm(LAC-LC)
    return AL, AR, errL, errR


# function min_AC_C(AC::Dims3Array,C::AbstractMatrix)
#     ## Algorithm5
#     D, d, = size(AC)
#     # F = qr(reshape(AC,(D*d, D))
#     # QAC,RAC = Matrix(F.Q), F.R
#     QAC, RAC = qrpos(reshape(AC,(D*d, D))) ## polar left for AC
#     # F = qr(C)
#     # QC,RC = Matrix(F.Q), F.R
#     QC, RC = qrpos(C) ## polar left for C
#     AL = reshape(QAC*QC', (D, d, D))
#     errL = norm(RAC-RC) ## not sure why
#     LAC, QAC = lqpos(reshape(AC,(D, d*D))) ## polar right for AC
#     LC, QC = lqpos(C) ## polar right for C
#     AR = reshape(QC'*QAC, (D, d, D))
#     errR = norm(LAC-LC) 
#     return AL, AR, errL, errR
# end


# def min_AC_C(AC, C):
#     ## algorithm 5
#     D, d = AC.shape()
#     QAC, RAC = qrpos(AC.reshape(D*d, D)) ## polar left for AC
#     QC, RC = qrpos(C)
#     AL = (QAC*QC).reshape(D, d, D)
#     errL = cy.linalg.Norm(RAC-RC) ## not sure why
#     LAC, QAC = lqpos(AC.reshape(D*d, D)) ## polar left for AC
#     LC, QC = lqpos(C)
#     AR = (QC*QAC).reshape(D, d, D)
#     errL = cy.linalg.Norm(LAC-LC)
#     return AL, AR, errL, errR


