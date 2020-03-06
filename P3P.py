import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """ 
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 3x1 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####
    a = np.linalg.norm(Pw[3,:] - Pw[2,:])
    b = np.linalg.norm(Pw[3,:] - Pw[1,:])
    c = np.linalg.norm(Pw[2,:] - Pw[1,:])
    
    f = (K[0,0] + K[1,1])/2
    cx,cy = K[0,2],K[1,2]
    
    u1,v1 = Pc[1,0]-cx, Pc[1,1]-cy
    u2,v2 = Pc[2,0]-cx, Pc[2,1]-cy
    u3,v3 = Pc[3,0]-cx, Pc[3,1]-cy

    j1 = np.array([u1,v1,f]) / np.sqrt(u1**2 + v1**2 + f**2)
    j2 = np.array([u2,v2,f]) / np.sqrt(u2**2 + v2**2 + f**2)
    j3 = np.array([u3,v3,f]) / np.sqrt(u3**2 + v3**2 + f**2)

    cosa = np.dot(j2,j3)
    cosb = np.dot(j1,j3)
    cosg = np.dot(j1,j2)

    # define coefficients of the 4th degree polynomial
    A = (a**2 - c**2)/b**2
    B = (a**2 + c**2)/b**2

    a4 = (A - 1)**2 - 4*c**2/b**2 * cosa**2
    
    a3 = 4*(A * (1 - A) * cosb -  (1 - B) * cosa * cosg + 2*c**2/b**2 * cosa**2 * cosb )
    
    a2 = 2*(A**2 - 1 + 2 * A**2 * cosb**2 + 2*(b**2 - c**2)/b**2 * cosa**2 - 4*B * cosa*cosb*cosg + 2*(b**2 - a**2)/b**2 * cosg**2)
    
    a1 = 4*(-A * (1 + A) * cosb + 2*a**2/b**2 * cosg**2 * cosb - (1 - B) * cosa * cosg )
    
    a0 = (1+ A)**2 - 4*a**2/b**2*cosg**2 

    poly_coeffs = [a4,a3,a2,a1,a0]

    # calculate real roots u and v
    roots_v = np.roots(poly_coeffs)

    real_v = roots_v[np.isreal(roots_v)].real
    real_u = ((-1 + A) * real_v**2 - 2*A*cosb*real_v +1 + A)/(2*cosg - 2*real_v*cosa)
    print(real_v)
    
    # check for distances
    for i,v in enumerate(real_v):
        
        u = real_u[i]
        d1 = a**2/(u**2 + v**2 - 2*u*v*cosa)
        d1 = np.sqrt(d1)
        d2 = u*d1
        d3 = v*d1
        if d1>0 and d2>0 and d3>0:
            break
    
    Pc_3d = np.zeros((3,3))
    Pc_3d[0,:] = j1*d1
    Pc_3d[1,:] = j2*d2
    Pc_3d[2,:] = j3*d3

    R,t = Procrustes(Pc_3d, Pw[1:4])
    
    return R, t

def Procrustes(X, Y):
    """ 
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate 
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 1x3 numpy array describing camera translation in the world (t_wc)
        
    """
    #
    a21 = Y[1,:] - Y[0,:]
    a31 = Y[2,:] - Y[0,:]
    a = np.cross(a21, a31)
    A = np.vstack([a21, np.cross(a, a21), a]).T
    
    b21 = X[1,:] - X[0,:]
    b31 = X[2,:] - X[0,:]
    b = np.cross(b21, b31)
    B = np.vstack([b21, np.cross(b, b21), b]).T
    
    #
    M = B @ A.T
    U, S, VT = np.linalg.svd(M)
    V = VT.T
    
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(V @ U.T)
    R = V @ d @ U.T
    t = Y.mean(axis=0) - R.dot(X.mean(axis=0)) 
    
    return R, t