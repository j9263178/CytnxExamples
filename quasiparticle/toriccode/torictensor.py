import numpy as np
from scipy.linalg import expm
from ncon import ncon

def mod(a, b):
    return a%b

def symstringtoric(bx, bz):
    sx = np.zeros([2, 2])
    sz = np.zeros([2, 2])
    sx[0,1] = sx[1,0] = 1
    sz[0,0] = 1; sz[1,1] = -1
    toric = np.zeros([2,2,2,2,2,2,2,2])
    ## See Schuch et al 2010 Annals of Physics 325 (2010) 2153–2192
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    toric[mod(i+j,2),mod(j+k,2), mod(k+l,2), mod(l+i,2), i,j,k,l] = 1
    toric = toric.reshape(16, 2, 2, 2, 2)
    ## Add string tension. See Haegeman et al 2015 10.1038/ncomms9284
    ST = expm(bz * sz / 4 + bx * sx / 4)
    ST_4 = ncon([ST,ST,ST,ST], [[-1, -5], [-2, -6], [-3, -7], [-4, -8]])
    ST_4 = ST_4.reshape(16, 16)
    toric = ncon([toric, ST_4], [[1, -2, -3, -4, -5], [1, -1]])
    ## Use Hadamard gate to transform X^4 invariant to Z^2 invariant
    H = np.zeros([2, 2])
    H[0, 0]= H[0, 1] = H[1, 0] = 1
    H[1, 1] = -1
    H *= 2**(0.5)
    toric = ncon([toric, H, H, H, H], [[-1,1,2,3,4], [1,-2],[2,-3],[3,-4], [4,-5]])
    return toric/np.max(np.abs(toric))


# function symstringtoric(βx, βz)
#     sx = zeros(2,2)
#     sz = zeros(2,2)
#     sx[1,2] = sx[2,1] = 1
#     sz[1,1] = 1; sz[2,2] = -1
#     toric = zeros(2,2,2,2,2,2,2,2) 
#     ## See Schuch et al 2010 Annals of Physics 325 (2010) 2153–2192
#     for i = 0:1, j = 0:1, k = 0:1, l = 0:1
#         toric[mod(i+j,2)+1,mod(j+k,2)+1, mod(k+l,2)+1, mod(l+i,2)+1, i+1,j+1,k+1,l+1] = 1
#     end
#     toric = reshape(toric, 16, 2, 2, 2, 2)
#     ## Add string tension. See Haegeman et al 2015 10.1038/ncomms9284
#     ST = exp(βz * sz / 4. + βx * sx / 4.)
#     ST_4 = ncon((ST,ST,ST,ST),
#                 ([-1, -5], [-2, -6], [-3, -7], [-4, -8]))
#     ST_4 = reshape(ST_4,16,16)
#     toric = ncon([toric, ST_4],
#             ([1, -2, -3, -4, -5], [1, -1]))
#     ## Use Hadamard gate to transform X^4 invariant to Z^2 invariant
#     H = sqrt(2)*[1. 1.; 1. -1.]
#     toric = ncon([toric, H, H, H, H], [[-1,1,2,3,4], [1,-2],[2,-3],[3,-4], [4,-5]])
#     return toric/maximum(abs.(toric))
# end


