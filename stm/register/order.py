import numpy as np

def sort_template(template, method='azimuth'):
    
    positions = template.positions
    
    if method == 'azimuth':
        permutation = azimuth_sort(positions)
    elif method == 'euclidean':
        permutation = euclidean_sort(positions)
    elif method == 'shell-azimuth':
        permutation = shell_azimuth_sort(positions)
    else:
        raise NotImplementedError()
        
    template.reorder(permutation)

def sort_segments(segments, method='azimuth'):
    
    for segment in segments:
        
        positions = segment.points.positions - segment.origin.position
        
        if method == 'azimuth':
            permutation = azimuth_sort(positions)
        elif method == 'euclidean':
            permutation = euclidean_sort(positions)
        elif method == 'shell-azimuth':
            permutation = shell_azimuth_sort(positions, [[0,1],[2,3],[4,5,6,7],[8,9]])
        else:
            raise NotImplementedError()
        
        segment.indices = [segment.indices[i] for i in permutation]
    
    return segments

def azimuth_sort(points):
    
    azimuth = np.arctan2(points[:,0], points[:,1])
    
    permutation = np.argsort(azimuth)
    
    return permutation

def euclidean_sort(points):
    
    norms = -np.linalg.norm(points, axis=1)
    
    permutation = np.argsort(norms)
    
    return permutation

def labels2groups(labels):

    labels = [[i,label] for i,label in enumerate(labels)]
    values = set(map(lambda x:x[1], labels))
    groups = [[y[0] for y in labels if y[1]==x] for x in values]
    
    return groups

def group_by_shell(norms, tol=1e-12):
    
    asort = np.argsort(norms)    
    edges = np.diff(norms[asort]) > tol
    labels = np.hstack((0,np.cumsum(edges)))
    labels = labels[np.argsort(asort)]
    groups = labels2groups(labels)
    
    return groups

def group_by_angle(angle, tol=1e-12):
    
    asort = np.argsort(angle)
    angle = np.hstack((angle[asort][-1], angle[asort]))
    labels = np.cumsum((np.diff(angle) % np.pi) > tol)-1
    labels = labels[np.argsort(asort)]
    groups = labels2groups(labels)
    
    return shells

def shell_azimuth_sort(positions, shells=None, roll=0, tol=1e-12):
    
    norms = np.linalg.norm(positions,axis=1)
    azimuth = np.arctan2(positions[:,0],positions[:,1])
    
    if shells is None:
        shells = group_by_shell(norms)
    
    last_azimuth=0
    for i,shell in enumerate(shells):
        
        shell_azimuth=azimuth[shell]
        
        azimuth_sort=np.argsort((shell_azimuth-last_azimuth)%(2*np.pi))
        
        if i == 0:
            azimuth_sort = np.roll(azimuth_sort, -roll)
            
        shells[i]=np.array(shell)[azimuth_sort]
        
        last_azimuth = shell_azimuth[azimuth_sort[-1]]
    
    permutation = np.hstack(shells)
    
    return permutation

def find_symmetry(points, tol=1e-12):

    norms = np.linalg.norm(points,axis=1)
    
    azimuth = np.arctan2(points[:,0], points[:,1])
    
    a_sort = np.argsort(azimuth)
    
    azimuth = azimuth[a_sort]
    norms = norms[a_sort]
    
    shells = group_by_shell(norms, tol=tol)

    azimuth = (azimuth + np.pi) % (2 * np.pi )
    
    symmetry = []
    for shell in shells:
        symmetry.append([])

        a = azimuth[shell]
        
        a=np.sort(a)
        a=np.hstack((a,a[0]+2*np.pi))

        d_azimuth=np.abs(np.diff(a))
        d_azimuth=np.hstack((d_azimuth,d_azimuth))
        
        for i in range(1,len(a)-1):
            if np.all(np.abs(d_azimuth[:len(a)]-np.roll(d_azimuth,i)[:len(a)]) < tol):
                symmetry[-1].append(np.sum(d_azimuth[:i]))

        symmetry[-1]=np.array(symmetry[-1])
    
    for i in range(0,len(symmetry)):
        symmetry[0] = symmetry[0][(np.abs(symmetry[i][:,None] - symmetry[0]) < tol).any(0)]
    
    if len(symmetry[0]) > 0 :
        order = int(np.round(2 * np.pi / symmetry[0][0]))
    else:
        order = 1
    
    return order
    
def generate_non_equivalent(points, symmetry_order, tol=1e-12):
    
    permutations=[]
    for i in range(0,len(points)//symmetry_order):
        
        permutation = np.arange(0,len(points),1)
        
        permutation = np.roll(permutation, -i)
        
        permutations.append(permutation)
    
    return permutations