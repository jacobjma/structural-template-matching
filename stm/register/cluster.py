import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from stm.register import match

def hierarchical_clustering(distance_matrix, threshold, criterion = 'distance'):
    
    if len(distance_matrix.shape) == 2:
        distance_matrix = squareform(distance_matrix)
    
    linkage = hierarchy.linkage(distance_matrix)
    clusters = hierarchy.fcluster(linkage, threshold, criterion=criterion)
    
    unique, counts = np.unique(clusters, return_counts=True)
    
    unique = unique[np.argsort(-counts)]
    
    reordered_clusters = np.zeros_like(clusters)
    for i,u in enumerate(unique):
        reordered_clusters[clusters == u] = i
    
    return reordered_clusters

def calc_distance_matrix(segments, scale_invariant=False, progress_bar=False):
    
    S = [s[1:] - s[0] for s in segments.segment_points]
    
    if scale_invariant:
        s = [normalize_points(s) for s in S]
    
    return match.match_self(s, progress_bar=progress_bar)

def cluster_points(points, max_dist, return_counts=False):
    
    clusters = fcluster(linkage(points), max_dist, criterion='distance')
    unique, indices = np.unique(clusters, return_index=True)
    points = np.array([np.mean(points[clusters==u],axis=0) for u in unique])
    
    if return_counts:
        counts = np.array([np.sum(clusters==u) for u in unique])
        return points, counts
    else:
        return points