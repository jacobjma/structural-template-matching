import numpy as np
from scipy.stats import trim_mean
from scipy.linalg import polar

def affine_transform(src, dst):
    
    coeffs = range(6)
    
    xs = src[:, 0]
    ys = src[:, 1]
    xd = dst[:, 0]
    yd = dst[:, 1]
    rows = src.shape[0]

    # params: a0, a1, a2, b0, b1, b2, c0, c1
    A = np.zeros((rows * 2, 9))
    A[:rows, 0] = xs
    A[:rows, 1] = ys
    A[:rows, 2] = 1
    A[:rows, 6] = - xd * xs
    A[:rows, 7] = - xd * ys
    A[rows:, 3] = xs
    A[rows:, 4] = ys
    A[rows:, 5] = 1
    A[rows:, 6] = - yd * xs
    A[rows:, 7] = - yd * ys
    A[:rows, 8] = xd
    A[rows:, 8] = yd

    # Select relevant columns, depending on params
    A = A[:, list(coeffs) + [8]]

    _, _, V = np.linalg.svd(A)

    H = np.zeros((3, 3))
    # solution is right singular vector that corresponds to smallest
    # singular value
    H.flat[list(coeffs) + [8]] = - V[-1, :-1] / V[-1, -1]
    H[2, 2] = 1

    return H

def calc_deformation(src, dst):
    
    A = affine_transform(src,dst)[:-1,:-1]
    U,P = polar(A,side='left')
    rotation = np.arctan2(U[1,0], U[0,0])
    rotation = (rotation + np.pi) % (2 * np.pi )
    
    return rotation, P

def calibrate_strain(strain, proportiontocut=None):

    planar_deform = 1 + (strain[:,0,0] + strain[:,1,1]) / 2
    
    if proportiontocut is None:
        mean_strain = np.nanmean(planar_deform)
    else:
        mean_strain = trim_mean(planar_deform[planar_deform.mask==0], proportiontocut)
    
    strain[:,0,0] = (1 + strain[:,0,0]) / mean_strain - 1
    strain[:,1,1] = (1 + strain[:,1,1]) / mean_strain - 1
    strain[:,0,1] = strain[:,0,1] / mean_strain 
    strain[:,1,0] = strain[:,1,0] / mean_strain
    
    return strain
    
def rotate_points(points, angle, center='cop'):
    """Rotate point positions.

    Parameters:
    angle = None:
        Angle that the points are rotated
    center = (0, 0, 0):
        The center is kept fixed under the rotation. Use 
        'COP' to fix the center of positions."""
    
    if isinstance(center, str):
        if center.lower() == 'cop':
            center = points.mean(axis=0)
    else:
        center = np.array(center)
    
    cosa = np.cos(angle) 
    sina = np.sin(angle) 
    
    R = np.array([[cosa,sina],[-sina,cosa]])

    return np.dot(points - center, R.T) + center

def rotate_strain(rotation, strain, proportiontocut=None):
    
    cosa = np.cos(rotation) 
    sina = np.sin(rotation) 
    
    R = np.array([[cosa, sina], [-sina, cosa]])
    
    mask = strain.mask.copy()
    
    mask_flat = np.all(np.all(strain.mask == True,axis=1),axis=1)
    
    strain[mask_flat] = np.array([np.dot(R, np.dot(s, R.T)) for s,m in zip(strain,mask_flat) if m])
    
    strain.mask = mask
    
    return strain
    