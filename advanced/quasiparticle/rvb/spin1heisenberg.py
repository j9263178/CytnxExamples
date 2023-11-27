
def spin1Heisenberg():

    J = 1
    
    sx = np.array([[0, 1, 0], [1, 0, 1],[0, 1, 0]])
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
    sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

    sp = sx+1j*sy
    sm = sx-1j*sy
    eye = np.eye(3)

    M = np.zeros([5, 5, 3, 3])
    M[0,0] = eye; M[4,4] = eye
    M[0,1] = sp;  M[1,4] = -J/2 * sm
    M[0,2] = sm;  M[2,4] = -J/2 * sp
    M[0,3] = sz;  M[3,4] = -Delta * J * sz
    M[0,4] = -h*sz

    ML = np.array([1,0,0,0,0]).reshape(5,1,1)
    MR = np.array([0,0,0,0,1]).reshape(5,1,1)
    return M, ML, MR