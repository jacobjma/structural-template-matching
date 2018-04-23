import numpy as np
import scipy.spatial
import sklearn.neighbors
from stm.segment import union_find, Segments

def simplex_edges(simplices):

    edges = []
    for s in simplices:
        for i in range(len(s)):
            a = s[i]
            b = s[(i + 1) % 3]
            edges += [(a, b)]
    return np.sort(np.array(edges))

def calc_circumcentre(p1, p2, p3):

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    a = np.linalg.det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    bx = -np.linalg.det([[x1**2 + y1**2, y1, 1],
                         [x2**2 + y2**2, y2, 1],
                         [x3**2 + y3**2, y3, 1]])

    by = np.linalg.det([[x1**2 + y1**2, x1, 1],
                        [x2**2 + y2**2, x2, 1],
                        [x3**2 + y3**2, x3, 1]])
    x0 = -bx / (2 * a)
    y0 = -by / (2 * a)
    return np.array([x0, y0])

def merge_simplices(simplices, vertices, threshold):

    num_simplices = len(simplices)
    ranks = np.zeros(num_simplices).astype(np.int)
    parents = np.arange(num_simplices)

    edges = [tuple(e) for e in simplex_edges(simplices)]
    nbrs = {e: [] for e in edges}
    for j, edge in enumerate(edges):
        nbrs[edge] += [j // 3]

    edges = np.array([v for v in nbrs.values() if len(v) == 2])
    
    pairs = vertices[edges]
    vecs = pairs[:, 1, :] - pairs[:, 0, :]
    lengths = np.linalg.norm(vecs, axis=1)

    indices = np.where(lengths < threshold)[0]
    for a, b in edges[indices]:
        union_find.find_and_merge(ranks, parents, a, b)

    merged = dict()
    for i in range(num_simplices):
        p = union_find.find(parents, i)
        if p not in merged:
            merged[p] = []
        merged[p] += [i]
    return merged

def order_exterior_vertices(simplices):

    edges = [tuple(e) for e in simplex_edges(simplices)]
    exterior = [e for e in set(edges) if edges.count(e) == 1]
    
    order = []
    (a, b) = exterior.pop(0)
    order += [a]
    while len(exterior):
        index = [i for i, (u, v) in enumerate(exterior) if u == b or v == b][0]
        u, v = exterior[index]
        if v == b:
            u, v = v, u
        exterior.pop(index)
        a, b = u, v
        order += [a]
    return order

def estimate_bond_length(positions, simplices):

    edges = simplex_edges(simplices)

    pairs = positions[edges]
    vecs = pairs[:, 1, :] - pairs[:, 0, :]
    lengths = np.linalg.norm(vecs, axis=1)

    nbrs = {i: [] for i in range(len(positions))}
    for (a, b), d in zip(edges, lengths):
        nbrs[a] += [d]

    lengths = [e for ds in nbrs.values() for e in sorted(ds)[:3]]
    return np.median(lengths)

def filtered_simplices(positions):

    simplices = scipy.spatial.Delaunay(positions).simplices
    vs = np.array([calc_circumcentre(*positions[s]) for s in simplices])
    
    x0 = np.min(positions[:, 0])
    x1 = np.max(positions[:, 0])
    y0 = np.min(positions[:, 1])
    y1 = np.max(positions[:, 1])

    indices = np.where((vs[:, 0] > x0) & (vs[:, 0] < x1) &
                       (vs[:, 1] > y0) & (vs[:, 1] < y1))[0]
    return simplices[indices], vs[indices]

def segment_holes(points, k=1.0):
    """Segments atomic positions of a graphene structure into holes.

    positions: The atomic positions.
    k: Merging factor (a larger factor results in more aggresive merging).

    Returns
    =======

    polygons: list of numpy arrays
        The vertices of each hole. The vertices are ordered in a path
        which represent the polygon exterior.
    """
    
    simplices, vertices = filtered_simplices(points)
    bond_length = estimate_bond_length(points, simplices)
    threshold = k * bond_length
    
    merged = merge_simplices(simplices, vertices, threshold)
    
    centers = np.zeros((len(merged),2))
    indices = []
    for i,v in enumerate(merged.values()):
        order = order_exterior_vertices(simplices[v])

        p = points[order]
        center = p.mean(axis=0)

        a = (p[0][0]-center[0])*(p[1][1]-center[1])
        b = (p[1][0]-center[0])*(p[0][1]-center[1])
        if a-b > 0:
            order = list(reversed(order))
                
        indices.append(order)
    
    return Segments(points, indices, origins='cop')

def segment_surrounding_holes(points, k=1.0):

    segments = segment_holes(points, k)
    
    new_indices = [set() for i in range(len(points))]
    
    for indices in segments.indices:
        for index in indices:
            new_indices[index].update([i for i in indices if i != index])
    
    new_indices = [[i]+list(j) for i,j in enumerate(new_indices)]
    
    new_indices = [i for i in new_indices if len(i)>1]
    
    segments = Segments(points, new_indices, 'front')
    
    return segments

def segment_neighbors(points, n_points, algorithm='ball_tree'):
    
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors = n_points, algorithm = algorithm)
    
    nn.fit(points)
        
    distances, indices = nn.kneighbors(points)
    
    return Segments(points, indices.tolist())