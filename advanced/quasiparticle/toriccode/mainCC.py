import numpy as np
from ncon import ncon
from vumpsfixedpoints import vumpsfixedpts
from excitationmpo import excitation
from torictensor import symstringtoric

########## Initial states
# βx, βz = 0.4, 0.8
# bx, bz = 0.4, 1.3 #d
bx, bz = 0.4, 1.1 #c
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
p_list = np.linspace(0, 1.0, 11)

ws = []
for p in p_list:
    data = excitation(W, AL, AR, C, FL, FR, num_w, p, charge = False, domain = False, Fstring = None, verbose = True)
    phi = data["phi"]; w = data["w"]
    ws.append(w)

np.savetxt(filename+'_II.txt', np.asarray(ws, dtype = np.float32))


ws = []
for p in p_list:
    data = excitation(W, AL, AR, C, FL, FR, num_w, p, charge = False, domain = True, Fstring = [IZ], verbose = True)
    phi = data["phi"]; w = data["w"]
    ws.append(w)

np.savetxt(filename+'_IZ.txt', np.asarray(ws, dtype = np.float32))


ws = []
for p in p_list:
    data = excitation(W, AL, AR, C, FL, FR, num_w, p, charge = False, domain = True, Fstring = [ZZ], verbose = True)
    phi = data["phi"]; w = data["w"]
    ws.append(w)
np.savetxt(filename+'_ZZ.txt', np.asarray(ws, dtype = np.float32))



# for p in p_list
#     data = excitation(W,AL,AR,C,FL,FR, num_ω, p; charge = false,domain = false, 
#     Fstring = Nothing, verbose = true)
#     ϕ = data["ϕ"]; ω = data["ω"]
#     # println(size(ω_Ceq))
#     len = length(ω)
#     open(folder_out*"$(filename)_II.txt", "a") do io
#         writedlm(io, reshape([p; ω[1:len]], 1,len+1))
#     end   
# end
# ##
# for p in p_list
#     data = excitation(W,AL,AR,C,FL,FR, num_ω, p; charge = false,domain = true, 
#     Fstring = IZ, verbose = true)
#     ϕ = data["ϕ"]; ω = data["ω"]
#     # println(size(ω_Ceq))
#     len = length(ω)
#     open(folder_out*"$(filename)_IZ.txt", "a") do io
#         writedlm(io, reshape([p; ω[1:len]], 1,len+1))
#     end   
# end
# ##
# for p in p_list
#     data = excitation(W,AL,AR,C,FL,FR, num_ω, p; charge = false,domain = true, 
#     Fstring = ZZ, verbose = true)
#     ϕ = data["ϕ"]; ω = data["ω"]
#     # println(size(ω_Ceq))
#     len = length(ω)
#     open(folder_out*"$(filename)_ZZ.txt", "a") do io
#         writedlm(io, reshape([p; ω[1:len]], 1,len+1))
#     end   
# end