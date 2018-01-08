import numpy as np

def calc_distance_squared(src, dst):
    return np.sum((src - dst)**2)

def calc_rotation(src, dst):
    A = np.dot(dst.T, src)
    
    V, S, W = np.linalg.svd(A)
    U = np.dot(V, W)
    return U

def rmsd_kabsch(src, dst, return_rotated=False):

    U = calc_rotation(src, dst)
    rotated = np.dot(src, U.T)
    rmsd = np.sqrt(calc_distance_squared(dst, rotated))
    
    if return_rotated:
        return rmsd, rotated
    else:
        return rmsd