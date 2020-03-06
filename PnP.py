import numpy as np
from est_homography import est_homography

def PnP(Pc, Pw, K=np.eye(3)):
    """ 
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic
    
    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 3x1 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    
    H = est_homography(Pw[:,:2], Pc)
    H = np.linalg.inv(K) @ H

    M = H.copy()
    M[:,2] = np.cross(M[:,0],  M[:,1])

    U, S, VT = np.linalg.svd(M)

    N = np.eye(3)
    N[2,2] = np.linalg.det(U @ VT)
    R = U @ N @ VT
    t = H[:,2] / np.linalg.norm(H[:,0], ord=2)

    # change from R_cw, t_cw to R_wc, t_wc
    t = -R.T.dot(t)
    R = R.T

    ##### STUDENT CODE END #####

    return R, t
