import numpy as np
from ncon import ncon
from vumpsfixedpoints import vumpsfixedpts
from excitationmpo import excitation
from torictensor import symstringtoric

########## Initial states
# βx, βz = 0.4, 0.8
# bx, bz = 0.6, 0.5 # e
bx, bz = 0.4, 0.8 # e'
# βx, βz = 0.8, 0.5

Dmps = 6 ## MPS's accuracy-controlled bond dimension

filename = "bx%.1fbz%.1fDmps%d"%(bx, bz, Dmps)
sI = np.eye(2)
sZ = np.zeros([2, 2])
sZ[0, 0] = 1; sZ[1, 1] = -1
## building blocks of string operators acting on both ket and bra layers
ZZ = ncon([sZ,sZ],[[-1,-3],[-2,-4]]).reshape(4, 4)
IZ = ncon([sI,sZ],[[-1,-3],[-2,-4]]).reshape(4, 4)
## get TC wavefunctions
toric = symstringtoric(bx, bz)
dim = toric.shape[1]
## Construct the double tensor
W = ncon([toric, toric.conj()], [[1, -1, -3, -5, -7], [1, -2, -4, -6, -8]])
d = dim**2
W = W.reshape(d, d, d, d)
A = np.random.rand(Dmps, d, Dmps)

######### Fixed points
la, AL, C, AR, FL, FR = vumpsfixedpts(A, W, steps = 10, tol = 1e-10)
# λ, AL, C, AR, FL, FR = vumpsfixedpts(AL, O; tol = 1e-10);
W /= la
########## Excitation
num_w = 15
num_p = 11
p_list = np.linspace(0, 1.0, num_p)
# open(folder_out*"$(filename)_triv_Ceq.txt", "w") do io
#     write(io, "# p w \n")
# end   
# open(folder_out*"$(filename)_triv_Cdiff.txt", "w") do io
#     write(io, "# p w \n")
# end   

# topologically trivial excitaitons
ceqs = np.zeros(num_p, num_w)
cdiffs = np.zeros(num_p, num_w)

for i in range(num_p):
    data = excitation(W, AL, AR, C, FL, FR, num_w, p_list[i], charge = True, Cstring = [ZZ], domain = False, Fstring = None, verbose = True)
    w_Ceq = data["w_Ceq"]
    w_Cdiff = data["w_Cdiff"]
    # len_Ceq = len(w_Ceq)
    # len_Cdiff = len(w_Cdiff)
    # println(size(w_Ceq))
    # print(len(w_Ceq), len(w_Cdiff))
    # if p == p_list[0]:
    #     len_Ceq = len(w_Ceq)-5
    #     len_Cdiff = len(w_Cdiff)-5
    ceqs[i, :len(w_Ceq)] = w_Ceq
    cdiffs[i, :len(w_Ceq)] = w_Cdiff

np.savetxt(filename+'_triv_Ceq.txt', np.asarray(ceqs, dtype = np.float32))
np.savetxt(filename+'_triv_Cdiff.txt', np.asarray(cdiffs, dtype = np.float32))

# topologically non-trivial excitaitons
ceqs = []
cdiffs = []
for p in p_list:
    data = excitation(W, AL, AR, C, FL, FR, num_w, p, charge = True, Cstring = [ZZ], domain = True, Fstring = [IZ], verbose = True)
    phi_Ceq = data["phi_Ceq"]; w_Ceq = data["w_Ceq"]
    phi_Cdiff = data["phi_Cdiff"]; w_Cdiff = data["w_Cdiff"]
    print(len(w_Ceq), len(w_Cdiff))
    # if p == p_list[0]:
    #     len_Ceq = len(w_Ceq)-5
    #     len_Cdiff = len(w_Cdiff)-5
    ceqs.append(w_Ceq[:2])
    cdiffs.append(w_Cdiff[:2])

np.savetxt(filename+'_domain_Ceq.txt', np.asarray(ceqs, dtype = np.float32))
np.savetxt(filename+'_domain_Cdiff.txt', np.asarray(cdiffs, dtype = np.float32))