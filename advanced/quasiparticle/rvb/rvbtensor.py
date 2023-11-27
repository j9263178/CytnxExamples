import numpy as np
from ncon import ncon

def prod(a):
    import operator
    from functools import reduce
    return reduce(operator.mul, a, 1)

def eijk(args):
    from math import factorial
    n = len(args)
    return prod(prod(args[j] - args[i] for j in range(i + 1, n)) / factorial(i) for i in range(n))

def zbar():
    z = np.zeros([3, 3])
    z[0, 0] = 1
    z[1, 1] = 1
    z[2, 2] = -1
    return z

def rvb():
    P = np.zeros([2, 3, 3])
    P[0, 0, 2] = P[0, 2, 0] = P[1, 1, 2] = P[1, 2, 1] = 1.
    epsilon = np.zeros([3, 3, 3])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                epsilon[i, j, k] = eijk((i,j,k))
                # epsilon[i, j, k] = eijk((i,j,k))*2**(-1/2)
    epsilon[2, 2, 2] = 1
    ePPP = ncon([epsilon, P, P, P], [[1, 2, 3], [-4,1,-1], [-5,2,-2], [-6,3,-3]]).reshape(3,3,3,8)
    ePPPe = ncon([ePPP, epsilon], [[-1, 1, -2, -5], [1, -4, -3]])
    return ePPPe/np.max(ePPPe)

def rvbPx(x):
    Px = np.zeros([2, 2, 3, 3])
    Px[0, 0, 0, 2] = Px[0, 0, 2, 0] = Px[0, 1, 1, 2] = Px[0, 1, 2, 1] = 1.
    Px[1, 0, 0, 2] = Px[1, 1, 1, 2] = x
    Px[1, 0, 2, 0] = Px[1, 1, 2, 1] = -x
    epsilon = np.zeros([3, 3, 3])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                epsilon[i, j, k] = eijk((i,j,k))
                # epsilon[i, j, k] = eijk((i,j,k))*2**(-1/2)
    epsilon[2, 2, 2] = 1
    P = Px.reshape(4, 3, 3)
    ePPP = ncon([epsilon, P, P, P], [[1, 2, 3], [-4,1,-1], [-5,2,-2], [-6,3,-3]]).reshape(3,3,3,64)
    ePPPe = ncon([ePPP, epsilon], [[-1, 1, -2, -5], [1, -4, -3]])
    return ePPPe/np.max(ePPPe)

def spins():
    sI = np.eye(3)
    sX = np.array([[0, 1, 0], [1, 0, 1],[0, 1, 0]])
    sY = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
    sZ = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    return sI, sX, sY, sZ

def AKLTsquare():
    bond = np.zeros([2, 2])
    bond[0, 1] = 2**(-1/2); bond[1, 0] = -2**(-1/2)
    I = np.eye(2)
    Y = np.zeros([2, 2], dtype = np.csingle)
    Y[0, 1] = -1; Y[1, 0] = 1
    P = np.zeros([5, 2, 2, 2, 2])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if i+j+k+l == 0:
                        P[0 ,i, j, k, l] = 1
                    elif i+j+k+l == 1:
                        P[1 ,i, j, k, l] = 4**(-1/2)
                    elif i+j+k+l == 2: 
                        P[2 ,i, j, k, l] = 6**(-1/2)
                    elif i+j+k+l == 3:  
                        P[3 ,i, j, k, l] = 4**(-1/2)  
                    elif i+j+k+l == 4:   
                        P[4 ,i, j, k, l] = 1        
    # aklt = ncon([P, Y, Y, I, I], [[-1, 1, 2, 3, 4],[1, -2], [2, -3], [3, -4], [4, -5]]) ### works
    aklt = ncon([P, bond, bond, bond, bond], [[-1, 1, 2, 3, 4],[1, -2], [2, -3], [3, -4], [4, -5]])
    aklt = ncon([P, Y, Y], [[-1, 1, 2, -4, -5],[1, -2], [2, -3]]) 
    return aklt


    