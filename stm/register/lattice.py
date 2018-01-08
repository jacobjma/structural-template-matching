import numpy as np
import collections

def convert_templates(templates, lengths):
    
    new_templates = []
    for template in templates:
        try:
            new_templates += template.as_static(lengths)
        except:
            new_templates.append(template)
    
    return new_templates

class DynamicTemplate(object):
    def __init__(self):
        pass

class RegularPolygon(DynamicTemplate):
    
    def __init__(self, sidelength=1):
        
        super().__init__()
        
        self.sidelength = sidelength
        
    def as_static(self, numpoints):
        
        if not isinstance(numpoints, collections.Iterable):
            numpoints = [numpoints]
        
        templates = []
        for n in numpoints:
            positions = np.zeros((n,2))
            
            L =  self.sidelength/(2*np.sin(np.pi/n))
            
            for i in range(n):
                positions[i,0]=np.cos(i*2*np.pi/n)
                positions[i,1]=np.sin(i*2*np.pi/n)
               
            templates.append(positions*L)
        
        return templates

def create_template(a, b, basis=[[0,0]], indices=None, n_points=None, center=(0,0)):
    
    a=np.array(a)
    b=np.array(b)
    basis=np.array(basis)
    
    positions=[]
    if (indices is None)&(not n_points is None):
        
        for i in range(-n_points,n_points+1):
            for j in range(-n_points,n_points+1):
                for p in basis:
                    positions.append((i*a+j*b)+p)
        
        norms = np.linalg.norm(positions,axis=1)
        positions=np.array(positions)[np.argsort(norms)][:n_points]
        
    elif (not indices is None)&(n_points is None):
        for hk in indices:
            for p in basis:
                positions.append((hk[0]*a+hk[1]*b)+p)
        
        positions = np.array(positions)
    else:
        RuntimeError('Set either indices or n_points')
    
    if isinstance(center, str):
        if center.lower() == 'cop':
            center = positions.mean(axis=0)
    else:
        center = center[0]*a + center[1]*b
        
    positions -= center
    
    return positions
    
def create_lattice(a, b, size, origin=(0,0), max_points=10**5):
    
    a=np.array(a)
    b=np.array(b)
    
    points=[]
    sublattice=[]
    
    m_ = 0
    old_len = -1
    while (old_len != len(points))&(len(points) < max_points):
        
        old_len = len(points)
        
        if m_ == 0:
            signs = [1]
        else:
            signs = [-1,1]
        
        for s in signs:
            m = s*m_
            
            if a[0] != 0.:
                lower_x = - m*b[0]/a[0]
            elif m*b[0] < 0:
                continue
            else:
                lower_x = -np.inf
                
            if a[1] != 0.:
                lower_y = - m*b[1]/a[1]
            elif m*b[1] < 0:
                continue
            else:
                lower_y = -np.inf
            
            lower = np.max((lower_x,lower_y))
            
            if a[0] != 0.:
                upper_x = -m*b[0]/a[0] + size[0]/a[0]
            elif m*b[0] > size[0]:
                continue
            else:
                upper_x = np.inf
            
            if a[1] != 0.:
                upper_y = -m*b[1]/a[1] + size[1]/a[1]
            elif m*b[1] > size[1]:
                continue
            else:
                upper_y = np.inf
            
            upper = np.min((upper_x,upper_y))
            
            if (lower <= upper):
                for n in np.arange(int(lower),int(upper)+1,1):
                    
                    points.append(n*a+m*b)
        
        m_+=1
    
    points = np.array(points) + origin
    
    return points