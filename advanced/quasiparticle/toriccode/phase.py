import numpy as np
from ncon import ncon
from vumpsfixedpoints import vumpsfixedpts
from excitationmpo import excitation
from torictensor import symstringtoric

########## Initial states

bxs = np.linspace(0, 1.5, 20)
bzs = np.linspace(0, 1.5, 20)

Dmps = 6 ## MPS's accuracy-controlled bond dimension

sI = np.eye(2)
sZ = np.zeros([2, 2])
sZ[0, 0] = 1; sZ[1, 1] = -1
sX = np.zeros([2, 2])
sX[0, 1] = 1; sX[1, 0] = 1
## building blocks of string operators acting on both ket and bra layers
ZZ = ncon([sZ,sZ],[[-1,-3],[-2,-4]]).reshape(4, 4)
IZ = ncon([sI,sZ],[[-1,-3],[-2,-4]]).reshape(4, 4)
XX = ncon([sX,sX],[[-1,-3],[-2,-4]]).reshape(4, 4)
IX = ncon([sI,sX],[[-1,-3],[-2,-4]]).reshape(4, 4)

expXXs = np.zeros([len(bxs), len(bzs)])
expIXs = np.zeros([len(bxs), len(bzs)])
for i in range(len(bxs)):
    for j in range(len(bzs)):
        print(i, j)
        toric = symstringtoric(bxs[i], bzs[j])
        W = ncon([toric, toric.conj()], [[1, -1, -3, -5, -7], [1, -2, -4, -6, -8]])
        dim = toric.shape[1]; d = dim**2
        W = W.reshape(d, d, d, d)
        A = np.random.rand(Dmps, d, Dmps)
        la, AL, C, AR, FL, FR = vumpsfixedpts(A, W, verbose = False, steps = 25, tol = 1e-8); W /= la
        AC = ncon([AL,C],[[-1,-2,3],[3,-3]])
        expXXs[i,j] = np.real(ncon([AC.conj(), XX, AC],[[3,1,4],[1,2],[3,2,4]]))
        expIXs[i,j] = np.real(ncon([AC.conj(), IX, AC],[[3,1,4],[1,2],[3,2,4]]))

np.save("expXXs_", expXXs)
np.save("expIXs_", expIXs)

# ########## Excitation
# num_w = 15
# p_list = np.linspace(0, 1.0, 11)
# # open(folder_out*"$(filename)_triv_Ceq.txt", "w") do io
# #     write(io, "# p w \n")
# # end   
# # open(folder_out*"$(filename)_triv_Cdiff.txt", "w") do io
# #     write(io, "# p w \n")
# # end   

# # topologically trivial excitaitons
# ceqs = []
# cdiffs = []

# for p in p_list:
#     data = excitation(W, AL, AR, C, FL, FR, num_w, p, charge = True, Cstring = [ZZ], domain = False, Fstring = None, verbose = True)
#     phi_Ceq = data["phi_Ceq"]; w_Ceq = data["w_Ceq"]
#     phi_Cdiff = data["phi_Cdiff"]; w_Cdiff = data["w_Cdiff"]
#     # len_Ceq = len(w_Ceq)
#     # len_Cdiff = len(w_Cdiff)
#     # println(size(w_Ceq))
#     # print(len(w_Ceq), len(w_Cdiff))
#     # if p == p_list[0]:
#     #     len_Ceq = len(w_Ceq)-5
#     #     len_Cdiff = len(w_Cdiff)-5
#     ceqs.append(w_Ceq[:2])
#     cdiffs.append(w_Cdiff[:2])
# np.savetxt(filename+'_triv_Ceq.txt', np.asarray(ceqs, dtype = np.float32))
# np.savetxt(filename+'_triv_Cdiff.txt', np.asarray(cdiffs, dtype = np.float32))