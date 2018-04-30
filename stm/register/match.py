import numpy as np
from itertools import combinations
from tqdm import tqdm
from stm.rmsd.kabsch import rmsd_kabsch
try:
    from stm.rmsd.qcp import rmsd_qcp
except:
    import warnings
    warnings.warn('Unable to import the fast C-version of the QCP-algorithm, using slow version.')
    from stm.rmsd.qcp_slow import rmsd_qcp
from stm.register import order, deform
from stm.register.bnb import bnb_search

def rms_points(points):
    return np.sqrt(np.sum(points ** 2) / points.shape[0])

def normalize_points(points):
    
    points -= np.mean(points,axis=0)
    
    rms = rms_points(points)
    
    points /= rms
    
    return points

def match_self(S, scale_invariant=True, progress_bar=False):
    
    S = [s[order.azimuth_sort(s)] for s in S]
    
    symmetries = [order.find_symmetry(s) for s in S]
    
    permutations = [order.generate_non_equivalent(s,n) for s,n in zip(S, symmetries)]
    
    if scale_invariant:
        S = [s / rms_points(s) for s in S]
    
    rmsd = np.empty((len(S),)*2)
    rmsd[:] = np.nan
    np.fill_diagonal(rmsd, 0)
    
    for i,s in enumerate(tqdm(S, disable=not progress_bar)):
    
        for j,(t,P) in enumerate(zip(S[:i], permutations)):
            best_rmsd = np.inf
            if len(s) == len(t):
                for p in P:
                    r = rmsd_kabsch(t[p], s) / np.sqrt(len(s))
                    
                    if r < best_rmsd:
                        best_rmsd = r
                        rmsd[i,j]=rmsd[j,i]=r
    
    return rmsd

def average_segment(S, num_iter=3, progress_bar=False):
    
    if not all(len(s) == len(S[0]) for s in S):
        raise RuntimeError()
    
    S = [s[order.azimuth_sort(s)] for s in S]
    
    symmetries = [order.find_symmetry(s) for s in S]
    
    permutations = [order.generate_non_equivalent(s,n) for s,n in zip(S, symmetries)]
    
    mean_segment = S[0]
    
    for i in range(num_iter):
        summed_segment = np.zeros_like(mean_segment)
        
        for s,P in tqdm(zip(S,permutations), disable=not progress_bar):
            best_rmsd = np.inf
            for p in P:
                r, rotated = rmsd_kabsch(s[p], mean_segment, return_rotated=True)
                
                if r < best_rmsd:
                    best_rmsd = r
                    best_segment = rotated
            
            summed_segment += best_segment
        
        mean_segment = summed_segment / len(S)
    
    return summed_segment / len(S)
    
def match_templates(S, T, method='angular-sort', scale_invariant=True, calc_strain=True, rmsd_max=np.inf, rmsd_algorithm='qcp', progress_bar=False):
    
    if method.lower() == 'angular-sort':
        S = [s[order.azimuth_sort(s)] for s in S]
        T = [t[order.azimuth_sort(t)] for t in T]
        symmetries = [order.find_symmetry(t) for t in T]
        permutations = [order.generate_non_equivalent(t,n) for t,n in zip(T, symmetries)]
    elif method.lower() == 'bnb':
        symmetries = [order.find_symmetry(t) for t in T]
        permutations = [[range(len(t)) for t in T]]
    
    if scale_invariant:
        scaled_T = [t/rms_points(t) for t in T]
    else:
        scaled_T = T
    
    rmsd = np.ma.masked_array(np.zeros(len(S), dtype=float), mask=True)
    
    template_index =  np.ma.masked_array(np.zeros(len(S), dtype=int), mask=True)
    
    strain =  np.ma.masked_array(np.zeros((len(S),2,2), dtype=float), mask=True)
    
    rotation =  np.ma.masked_array(np.zeros(len(S), dtype=float), mask=True)
    
    for i,s in enumerate(tqdm(S, disable=not progress_bar)):
        best_rmsd = np.inf
        
        if scale_invariant:
            scaled_s = s / rms_points(s)
        else:
            scaled_s = s
        
        
        for j, (t, scaled_t, P) in enumerate(zip(T, scaled_T, permutations)):
            
            if len(s) == len(t):
                for p in P:
                    print(p)
                    if method == 'bnb':
                        r, level, p, num_eval = bnb_search(scaled_t, scaled_s)
                    elif method == 'angular-sort':
                        if rmsd_algorithm is 'kabsch':
                            r = rmsd_kabsch(scaled_t[p], scaled_s) / np.sqrt(len(scaled_s))
                        elif rmsd_algorithm is 'qcp':
                            r = rmsd_qcp(scaled_t[p].astype(np.float), scaled_s.astype(np.float))
                        else:
                            raise NotImplementedError()
                    
                    if (r < best_rmsd) & (r < rmsd_max):
                        best_rmsd = r
                        best_t = t[p]
                        rmsd[i] = r
                        template_index[i] = j
        
        if calc_strain&(not rmsd.mask[i]):
            r, P = deform.calc_deformation(best_t, s)
            strain[i,0,0] = P[0,0] - 1
            strain[i,1,1] = P[1,1] - 1
            strain[i,1,0] = strain[i,0,1] = P[1,0]
            
            rotation[i] = r % (2*np.pi/np.max(symmetries))
    
    if calc_strain:
        return rmsd, template_index, strain, rotation
    else:
        return rmsd, template_index
        