import numpy as np

def rmsd_qcp(A, B):
    
    n_atoms = len(A)

    G_A = np.sum(A**2)
    G_B = np.sum(B**2)
    
    M = np.dot(B.T, A)
    
    Sxx, Sxy = M[0, :]
    Syx, Syy = M[1, :]
    
    Sxx2 = Sxx * Sxx
    Syy2 = Syy * Syy
    Sxy2 = Sxy * Sxy
    Syx2 = Syx * Syx
    
    C2 = -2.0 * (Sxx2 + Syy2 + Sxy2 + Syx2)
    
    SxypSyx = Sxy + Syx
    SxymSyx = Sxy - Syx
    SxxpSyy = Sxx + Syy
    SxxmSyy = Sxx - Syy
    Syy2mSxx2 = Syy2 - Sxx2
    Sxy2mSyx2 = Sxy2 - Syx2
    
    C0 = Sxy2mSyx2 * Sxy2mSyx2 + Syy2mSxx2 * Syy2mSxx2 \
        + (SxymSyx * SxxmSyy * SxymSyx * SxxmSyy) \
        + (SxypSyx * SxxpSyy * SxypSyx * SxxpSyy)

    E0 = (G_A + G_B) / 2.0
    
    C3 = C2**2-4*C0

    max_eigenvalue = np.sqrt(np.sqrt(C3)-C2)/np.sqrt(2)
    rmsd = np.sqrt(np.abs(2.0 * (E0 - max_eigenvalue) / n_atoms))
    
    return rmsd
    