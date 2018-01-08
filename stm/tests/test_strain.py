import numpy as np
from scipy.special import erf
from scipy.interpolate import griddata
from stm.register import match, lattice
from stm.segment.segmentation import segment_neighbors

def continuum_strain(points, W, A):
    
    x = points[:,0]
    y = points[:,1]
    
    w = 1 / W**2 / 2

    exx = A * (x + y) * np.exp(-w * (x**2 + y**2))
    eyy = A * (x - y) * np.exp(-w * (x**2 + y**2))

    exy = A / 2 * \
        ((np.exp(-w * y**2) * (np.sqrt(np.pi)/(2 * np.sqrt(w)) - 
        np.sqrt(np.pi) * np.sqrt(w) * y**2) * erf(np.sqrt(w) * x) + 
        y * np.exp(w * (-x**2 - y**2))) +
        (np.exp(-w * x**2) * (np.sqrt(np.pi)/(2 * np.sqrt(w)) -
        np.sqrt(np.pi) * np.sqrt(w) * x**2) * erf(np.sqrt(w) * y) - 
        x * np.exp(w * (-x**2 - y**2))))

    return exx, eyy, exy

def displacements(points, W, A):
    
    new_points = points.copy()
    
    x = points[:,0]
    y = points[:,1]
    
    w = 1 / W**2 / 2
    
    new_points[:,0] += A * (-np.exp(w * (-x**2 - y**2))/(2 * w) + 
                    (np.exp(-w * y**2) * np.sqrt(np.pi) * 
                    y * erf(np.sqrt(w) * x))/(2 * np.sqrt(w)))

    new_points[:,1] += A * (np.exp(w * (-x**2 - y**2))/(2 * w) +
                    (np.exp(-w * x**2) * np.sqrt(np.pi) * 
                    x * erf(np.sqrt(w) * y))/(2 * np.sqrt(w)))
    
    return new_points

    
W=20
A=0.001
n=5
shape=(100,)*2
tol=1e-4
    
a = np.array([0, 1])
b = np.array([1, 0])
#b = np.array([np.sin(2/3*np.pi), np.cos(2/3*np.pi)])

points = lattice.create_lattice(a, b, size=shape, origin=(0,0)) - shape[0]/2

displaced_points = displacements(points, W, A)

segments = segment_neighbors(displaced_points, n_points = n)

template = lattice.create_template(a, b, n_points = n)

segments.match(template, rmsd_max=.1)

rmsd = segments.rmsd
strain = segments.strain

exx, eyy, exy = continuum_strain(points, W, A)

exx_err = np.abs(exx[rmsd.mask==0]-strain[rmsd.mask==0][:,0,0])
eyy_err = np.abs(eyy[rmsd.mask==0]-strain[rmsd.mask==0][:,1,1])
exy_err = np.abs(exy[rmsd.mask==0]-strain[rmsd.mask==0][:,1,0])

assert np.mean(exx_err) < tol
assert np.mean(eyy_err) < tol
assert np.mean(exy_err) < tol
