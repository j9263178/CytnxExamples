import numpy as np
from ncon import ncon
from vumpsfixedpoints import vumpsfixedpts
from excitationmpo import excitation
from torictensor import symstringtoric

########## Initial states

# bx, bz = 1.0, 0.5 #g
bx, bz = 1.3, 0.5 #h

Dmps = 6

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
la, AL, C, AR, FL, FR = vumpsfixedpts(A, W, steps = 30, tol = 1e-10)
# λ, AL, C, AR, FL, FR = vumpsfixedpts(AL, O; tol = 1e-10);
W /= la
########## Excitation
num_w = 15
p_list = np.linspace(0, 1.0, 11)

wcee = []
wceo = []
wcoo = []
for p in p_list:
    data = excitation(W,AL,AR,C,FL,FR, num_w, p, charge = True, Cstring = [ZZ,IZ],domain = False, Fstring = None, verbose = True)
    w_Cee = data["w_Cee"]; w_Ceo = data["w_Ceo"]; w_Coo = data["w_Coo"]
    print("Cee : %d, Ceo : %d, Coo : %d"%(len(w_Cee), len(w_Ceo), len(w_Coo)))
    if p == p_list[0]:
        len_Cee = len(w_Cee)
        len_Ceo = len(w_Ceo)-2
        len_Coo = len(w_Coo)
    wcee.append(w_Cee[:3]); wceo.append(w_Ceo[:8]); wcoo.append(w_Coo[:2])

np.savetxt(filename+'_Cee.txt', np.asarray(wcee, dtype = np.float32))
np.savetxt(filename+'_Ceo.txt', np.asarray(wceo, dtype = np.float32))
np.savetxt(filename+'_Coo.txt', np.asarray(wcoo, dtype = np.float32))