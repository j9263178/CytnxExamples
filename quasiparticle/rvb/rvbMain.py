import numpy as np
from ncon import ncon
from vumpsfixedpoints import vumpsfixedpts
from excitationmpo import excitation
# from rvbexcitation import *
from scipy.sparse.linalg import LinearOperator, eigs
from rvbtensor import rvb, rvbPx, spins, zbar, AKLTsquare

"""
FL--     AC    --FR
         1
         |
FL-- 0---W---2 --FR
         |
FL--     3     --FR
"""


# def overlap(A, W):
#     D = A.shape[0]
#     la, AL, C, AR, _, _ = vumpsfixedpts(A, W, steps = 18, tol = 4.0e-4)
#     W = np.transpose(W, (2,3,0,1))
#     la, AL_, C_, AR_, _, _ = vumpsfixedpts(A, W, steps = 18, tol = 4.0e-4)
#     def rcon(v):
#         return ncon([v.reshape(D, D), AR, AR_.conj()], [[1,2], [-1,3,1], [-2,3,2]]).flatten()
#     def lcon(v):
#         return ncon([v.reshape(D, D), AL, AL_.conj()], [[1,2], [1,3,-1], [2,3,-2]]).flatten()
#     lop = LinearOperator((D*D, D*D), matvec=lcon)
#     rop = LinearOperator((D*D, D*D), matvec=rcon)
#     el, FL = eigs(lop, k = 1, which = 'LM', tol = 1e-8)
#     er, FR = eigs(rop, k = 1, which = 'LM', tol = 1e-8)
#     FL/=el
#     FR/=er
#     FR = FR.reshape(D, D)
#     FL = FL.reshape(D, D)
#     return ncon([FL ,C, C_.conj(), FR], [[1, 2],[1, 3],[2, 4],[3, 4]])

# #### rvb to dimer
# P, epsilon = rvbPx(0)
# ePPP = ncon([epsilon, P, P, P], [[1, 2, 3], [-4,1,-1], [-5,2,-2], [-6,3,-3]]).reshape(3,3,3,64)
# ePPPe = ncon([ePPP, epsilon], [[-1, 1, -2, -5], [1, -4, -3]])
# W = ncon([ePPPe, ePPPe.conj()],[[-1,-3,-5,-7,1],[-2,-4,-6,-8,1]]).reshape(9, 9, 9, 9)
# # W = np.transpose(W, (1,0,2,3))
# W = np.transpose(W, (2,3,0,1))


# ##### RVB virtual excitation
Dmps = 10
ePPPe = rvb()
sI, sX, sY, sZ = spins()
zbar = zbar()
ZZ = ncon([zbar,zbar],[[-1,-3],[-2,-4]]).reshape(9, 9)
IZ = ncon([sI,zbar],[[-1,-3],[-2,-4]]).reshape(9, 9)
W = ncon([ePPPe, ePPPe.conj()],[[-1,-3,-5,-7,1],[-2,-4,-6,-8,1]]).reshape(9, 9, 9, 9)
W = np.transpose(W, (1,2,3,0))
A = np.random.rand(Dmps, 9, Dmps)
la, AL, C, AR, FL, FR = vumpsfixedpts(A, W, steps = 200, tol = 1e-7)
W /= la
num_w = 15
num_p = 11
p_list = np.linspace(0, 1.0, num_p)

# topologically trivial excitaitons
ceqs = np.zeros([num_p, num_w])
cdiffs = np.zeros([num_p, num_w])

for i in range(num_p):
    data = excitation(W, AL, AR, C, FL, FR, num_w, p_list[i], charge = True, Cstring = [ZZ], domain = False, Fstring = None, verbose = True)
    w_Ceq = data["w_Ceq"]
    w_Cdiff = data["w_Cdiff"]
    ceqs[i, :len(w_Ceq)] = w_Ceq
    cdiffs[i, :len(w_Cdiff)] = w_Cdiff

np.savetxt('triv_Ceq.txt', np.asarray(ceqs, dtype = np.float32))
np.savetxt('triv_Cdiff.txt', np.asarray(cdiffs, dtype = np.float32))


# ws = []
# for p in p_list:
#     data = excitation(W, AL, AR, C, FL, FR, num_w, p, charge = True, Cstring = [ZZ], domain = False, Fstring = None, verbose = True)
#     w = data["w"]
#     w_Ceq = data["w_Ceq"]
#     w_Cdiff = data["w_Cdiff"]
#     ws.append(w)
# np.savetxt('rvb_p_w.txt', np.asarray(ws, dtype = np.float32))

# ws = []
# for p in p_list:
#     data = excitation(W,AL,AR,C,FL,FR, num_w, p, charge = False, Cstring = None, domain = True, Fstring = [IZ], verbose = True)
#     w = data["w"]
#     ws.append(w)
# np.savetxt('rvb_p_w_domain.txt', np.asarray(ws, dtype = np.float32))

# ####### correlation function
# Dmps = 30
# xs = np.linspace(0, 1, 11)
# xcr = []
# for x in xs:
#     print("x = ", x)
#     ePPPe = rvbPx(x)
#     W = ncon([ePPPe, ePPPe.conj()],[[-1,-3,-5,-7,1],[-2,-4,-6,-8,1]]).reshape(9, 9, 9, 9)
#     W = np.transpose(W, (2,3,0,1))
#     A = np.random.rand(Dmps, 9, Dmps)
#     la, AL, C, AR, FL, FR = vumpsfixedpts(A, W, steps = 80, tol = 4.0e-6)
#     A = np.random.rand(Dmps, 9, Dmps)
#     W = np.transpose(W, (2,3,0,1))
#     la, AL_, C_, AR_, FL, FR = vumpsfixedpts(A, W, steps = 80, tol = 4.0e-6)
#     # print(np.linalg.norm(ncon([AR, AR.conj()], [[-1, 2, 1],[-2, 2, 1]])-np.eye(Dmps)))
#     # exit(0)
#     AC = ncon([AL,C],[[-1,-2,3],[3,-3]])
#     T = ncon([AC.conj(), ZZ, AC],[[1,2,-1], [2,3], [1,3,-2]])
#     norm = (np.real(ncon([AC.conj(), AC],[[6,1,4], [6,1,4]])))
#     bias = (np.real(ncon([AC.conj(), ZZ, AC],[[6,1,4],[1,2],[6,2,4]]))**2)/norm**2
#     # bias = 0
#     # TI = ncon([AC.conj(), AC],[[1,2,-1], [1,2,-2]])
#     crs = []
#     # crs.append(np.real(ncon([AC.conj(), ZZ, ZZ, AC],[[6,1,4],[1,2],[2,3],[6,3,4]]))-bias)
#     crs.append(0.5)
#     for r in range(5):
#         # cr = np.real(ncon([T, AR.conj(), ZZ, AR],[[5,6], [5,1,4],[1,2],[6,2,4]]))-np.real(ncon([T, AR.conj(), AR],[[5,6], [5,2,4], [6,2,4]]))*np.real(ncon([TI, AR.conj(), ZZ, AR],[[5,6], [5,1,4],[1,2],[6,2,4]]))
#         cr = np.abs(np.real(ncon([T, AR.conj(), ZZ, AR],[[5,6], [5,1,4],[1,2],[6,2,4]])))/norm-bias
#         crs.append(cr)
#         T = ncon([T, AR.conj(), AR],[[5,6], [5,1,-1],[6,1,-2]])
#         # TI = ncon([TI, AR.conj(), AR],[[5,6], [5,1,-1],[6,1,-2]])
#     xcr.append(crs)
# np.save("xcr_D30_test", np.asarray(xcr))

# ######### gap of the transfer operator
# Dmps = 40
# Nh = 4
# xs = np.linspace(0, 1, 11)
# xcr = []
# for x in xs:
#     print("x = ", x)
#     P, epsilon = rvbPx(x)
#     ePPP = ncon([epsilon, P, P, P], [[1, 2, 3], [-4,1,-1], [-5,2,-2], [-6,3,-3]]).reshape(3,3,3,64)
#     ePPPe = ncon([ePPP, epsilon], [[-1, 1, -2, -5], [1, -4, -3]])
#     W = ncon([ePPPe, ePPPe.conj()],[[-1,-3,-5,-7,1],[-2,-4,-6,-8,1]]).reshape(9, 9, 9, 9)
#     # W = np.transpose(W, (1,0,2,3))
#     # W = np.transpose(W, (2,3,0,1))
#     # phy = 3
#     # for i in range(Nh):
#     #     phy *= 3
#     #     Tsf = ncon([Tsf, W],[[-1,-2,1,-5],[1,-3,-4,-6]]).reshape(Dmps, phy, Dmps, phy)
#     # print(Tsf.shape)
#     # # w, v = eig(Tsf)
    
# ########## change of overlap
# Dmps = 40
# dx = 0.01
# xs = np.linspace(0, 1, 11)
# A = np.random.rand(Dmps, 9, Dmps)
# fs = []
# W12s = []
# W11s = []
# W22s = []
# datas = []
# for x in xs:
#     print("x = ", x)
#     ePPPex = rvbPx(x)
#     ePPPedx= rvbPx(x+dx)
#     W11 = ncon([ePPPex, ePPPex.conj()],[[-1,-3,-5,-7,1],[-2,-4,-6,-8,1]]).reshape(9, 9, 9, 9)
#     W22 = ncon([ePPPedx, ePPPedx.conj()],[[-1,-3,-5,-7,1],[-2,-4,-6,-8,1]]).reshape(9, 9, 9, 9)
#     W12 = ncon([ePPPex, ePPPedx.conj()],[[-1,-3,-5,-7,1],[-2,-4,-6,-8,1]]).reshape(9, 9, 9, 9)
#     w12 = overlap(A, W12)
#     w11 = overlap(A, W11)
#     w22 = overlap(A, W22)
#     W12s.append(w12)
#     W11s.append(w11)
#     W22s.append(w22)
#     fs.append((1/dx**2)*(1 - w12**2/(w11*w22)))
# datas.append(fs)
# datas.append(W12s)
# datas.append(W11s)
# datas.append(W22s)
# np.save("fs",np.asarray(datas))

##### AKLT

# Dmps = 12
# aklt = AKLTsquare()
# W = ncon([aklt, aklt.conj()],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]).reshape(4, 4, 4, 4)

# A = np.random.rand(Dmps, 4, Dmps)
# la, AL, C, AR, FL, FR = vumpsfixedpts(A, W, steps = 100, tol = 1e-7)
# W /= la

# num_w = 30
# num_p = 11
# p_list = np.linspace(0, 1.0, num_p)

# pipws = np.ones([num_p, num_w])
# zeropws = np.ones([num_p, num_w])

# for i in range(num_p):
#     p = p_list[i]
#     D = AL.shape[0]
#     VL, nL = getnullspace(AL)
#     def operator(X):
#         B = ncon([VL, X.reshape(nL, D)],[[-1,-2,1],[1,-3]])
#         By = applyHeff(B, p*np.pi, AL, AR, C, W, FL, FR, pinv = "manual")   
#         Heff_X = ncon([By, VL.conj()],
#                     [[1,2,-2],[1,2,-1]])
#         return Heff_X
#     ws, excits = eigs(LinearOperator((D*nL, D*nL), matvec=operator, dtype=np.csingle), v0 = np.random.rand(nL, D), k = num_w)
#     phi = np.angle(ws)/np.pi
#     abs_ws = np.abs(ws)
#     # data = {}
#     # data["p"] = p; data["phi"] = phi[:num_w]; data["w"] = abs_ws[:num_w]
#     for j in range(num_w):
#         if np.abs(phi[j]) < 1e-6:
#             zeropws[i, j] = abs_ws[j]
#         else:
#             pipws[i, j] = abs_ws[j]
            
#     print("p = %.6e,  ω =  %.6e, ϕ = %.6e"%(p, abs_ws[0], phi[0]))

# np.savetxt('aklt_p_w_pi.txt', np.asarray(pipws, dtype = np.float32))
# np.savetxt('aklt_p_w_zero.txt', np.asarray(zeropws, dtype = np.float32))