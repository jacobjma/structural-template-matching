import numpy as np
import numbers
import collections
import scipy.stats
from stm.register import match, lattice, deform, cluster

def center_segment(segment, origin):
    if origin.lower() == 'front':
        segment = segment[1:] - segment[0]
    elif origin.lower() == 'cop':
        segment = segment - np.mean(segment,axis=0)
    return segment

class Segments(object):
    
    """Segments object.

    The Segments object can represent segments of a larger group of 
    points. It containes the original collection of points, and a list
    of lists of indices belonging to each segment. The origin of the 
    segments should 
    
    Per-segment properties (e.g. RMSD and strain) are stored in a dict
    of ndarrays.
     
    Parameters:
    ----------
    points: list of xy-positions
        Anything that can be converted to an ndarray of shape (n, 2) 
        will do: [(x1,y1), (x2,y2), ...].
    indices: list of list of ints
        Each segment is defined by the ints in a sublist.
    origins: str
        The origin of the segments can be chosen as
        'front': First listed index of the segment 
        'cop': Center of positions of the points in the segment
    """
    
    def __init__(self, points, indices, origins = 'front'):
        self._points = points.astype(np.float64)
        self._indices = indices
        self._origins = origins
        self._arrays = {}
        
    def __len__(self):
        return len(self._indices)
    
    @property
    def points(self):
        return self._points
    
    @property
    def indices(self):
        return self._indices
    
    @property
    def lengths(self):
        return [len(i) for i in self.indices]
    
    @property
    def segment_points(self):
        return [self.points[i] for i in self.indices]
    
    @property
    def best_rmsd(self):
        return np.min(self.rmsd,axis=1)
    
    @property
    def best_match(self):
        return np.argmin(self.rmsd,axis=1)
    
    @property
    def origins(self):
        if self._origins == 'front':
            return np.array([self.points[i][0] for i in self.indices])
        elif self._origins == 'cop':
            return np.array([np.mean(self.points[i],axis=0) for i in self.indices])
        else:
            raise NotImplementedError()
    
    def sample(self, f):
        N = np.round(len(self) * f).astype(int)
        return self[np.random.choice(len(self), N, replace=False)]
    
    def match(self, templates, method='angular-sort', scale_invariant=True, calc_strain=True, rmsd_max=np.inf, progress_bar=False, rmsd_algorithm='qcp'):
        
        """ Match segments to templates.
        
        Parameters:
        ----------
        templates: list of xy-positions or list of lists of xy-positions
            For each template anything that can be converted to an ndarray 
            of shape (n, 2) will do: [(x1,y1), (x2,y2), ...]. 
            If multiple templates are required use a list such arrays etc.
        method: 'str'
            The current methods for matching are:
            'angular-sort': Angular sorting using symmetries
            'bnb': Branch and bound search
        scale_invariant: bool
            Should the RMSD calculation take scale into account
        calc_strain: bool
            Should the strain calculation after matching 
        """
        
        T = templates
        
        if not isinstance(T, list):
            T = [T]
        
        T = lattice.convert_templates(T, set(self.lengths))
        
        T = [center_segment(t, self._origins) for t in T]
        
        S = [center_segment(s, self._origins) for s in self.segment_points]
        
        rmsd, template_index, strain, rotation = match.match_templates(S, T, method=method,
                    calc_strain=calc_strain, scale_invariant=scale_invariant, 
                    progress_bar=progress_bar, rmsd_max=rmsd_max, rmsd_algorithm=rmsd_algorithm)
        
        self.set_array('rmsd', rmsd)
        self.set_array('strain', strain)
        self.set_array('rotation', rotation)
        self.set_array('template_index', template_index)
    
    def calibrate_strain(self, proportiontocut):
        strain = deform.calibrate_strain(self.strain, proportiontocut)
        self.set_array('strain', strain)
    
    def calibrate_direction(self, proportiontocut, rotate_points=False):
        
        rotation = self.rotation
        
        if proportiontocut is None:
            mean_rotation = np.nanmean(rotation)
        else:
            mean_rotation = scipy.stats.trim_mean(rotation, proportiontocut)
        
        self.set_array('strain', deform.rotate_strain(mean_rotation, self.strain))
        
        self.set_array('rotation', rotation - mean_rotation)
        
        if rotate_points:
            self._points = deform.rotate_points(self._points, mean_rotation)
        
    def calc_distance_matrix(self, scale_invariant=True, progress_bar=True):
        
        S = [center_segment(s, self._origins) for s in self.segment_points]
        
        distance_matrix = match.match_self(S, scale_invariant=scale_invariant, progress_bar=progress_bar)
        
        self.set_array('distance_matrix', distance_matrix)
    
    def calc_clusters(self, threshold, criterion='distance'):
        
        clusters = cluster.hierarchical_clustering(self.distance_matrix, threshold, criterion)
        
        self.set_array('clusters', clusters)
    
    def average_segment(self, cluster=None):
        
        if cluster is None:
            S = self.segment_points
        else:
            indices = self.clusters == cluster
            S = [self.segment_points[j] for j,i in enumerate(indices) if i]
        
        S = [center_segment(s, self._origins) for s in S]
        
        average_segment = match.average_segment(S)
        
        if self._origins == 'front':
            average_segment = np.vstack(([0.,0.], average_segment))
        
        return average_segment
        
    def new_array(self, name, a, shape=None):
        """Add new array."""
        
        if name in self._arrays:
            raise ValueError('Array {} already exists'.format(name))
        
        if len(a) != len(self):
            raise ValueError('Array has wrong length: {} != {}.'.format(
                        len(a), len(self)))

        if shape is not None and a.shape[1:] != shape:
            raise ValueError('Array has wrong shape {} != {}.'.format(
                        a.shape, a.shape[0,:1]+shape))

        self._arrays[name] = a
    
    def get_array(self, name, copy=True):
        """Get an array.

        Returns a copy unless the optional argument copy is false."""
        
        if copy:
            return self._arrays[name].copy()
        else:
            return self._arrays[name]
    
    def set_array(self, name, a, shape=None):
        """Update array.

        If *shape* is not *None*, the shape of *a* will be checked.
        If *a* is *None*, then the array is deleted."""
        
        b = self._arrays.get(name)
        if b is None:
            if a is not None:
                self.new_array(name, a, shape)
        else:
            if a is None:
                del self._arrays[name]
            else:
                #a = np.asarray(a)
                if a.shape != b.shape:
                    raise ValueError('Array has wrong shape {} != {}.'.
                        format(a.shape, b.shape))
                
                b[:] = a
    
    def copy(self):
        """Return a copy."""
        
        indices = [i[:] for i in self.indices]
        
        segments = self.__class__(self.points, indices, origins=self._origins)
        
        for name, a in self._arrays.items():
            segments._arrays[name] = a.copy()
        
        return segments
    
    def extend(self, other):
        """Extend segments object by appending segments from *other*."""
        
        self._indices += other._indices
        
        n1 = len(self)
        n2 = len(other)

        for name, a1 in self._arrays.items():
            a = np.zeros((n1 + n2,) + a1.shape[1:], a1.dtype)
            a[:n1] = a1
            a2 = other._arrays.get(name)
            if a2 is not None:
                a[n1:] = a2
            self._arrays[name] = a

        for name, a2 in other._arrays.items():
            if name in self._arrays:
                continue
            a = np.empty((n1 + n2,) + a2.shape[1:], a2.dtype)
            a[n1:] = a2
            a[:n1] = 0

            self.set_array(name, a)
        
        return self
    
    def __add__(self, other):
        segments = self.copy()
        segments += other
        return segments
    
    __iadd__ = extend
    
    def __getattr__(self, name):
        """Get name attribute, return None if not explicitely set."""
        
        if name in self._arrays:
            return self._arrays[name]
        else:
            super().__getattribute__(name)
    
    def __getitem__(self, i):
        """Return a subset of the segments.

        i -- scalar integer, list of integers, or slice object
        describing which atoms to return.

        If i is a scalar, return a Segment object. If i is a list or a
        slice, return a Segments object with the same associated info 
        as the original Segments object.
        """
        
        if isinstance(i, numbers.Integral):
            natoms = len(self)
            if i < -natoms or i >= natoms:
                raise IndexError('Index out of range.')

            return Segment(segments=self, index=i)
        
        if isinstance(i, slice):
            indices = self.indices[i]
        else:
            indices = [self.indices[j] for j in i]
        
        segments = self.__class__(self.points, indices)

        segments._arrays = {}
        for name, a in self._arrays.items():
            segments._arrays[name] = a[i]
        
        return segments
    
    def __delitem__(self, i):
        
        if isinstance(i, numbers.Integral):
            del self._indices[i]
        else:
            
            for j in sorted(i, reverse=True):
                del self._indices[j]
        
            for name, a in self._arrays.items():
                np.delete(self._arrays[name], i, axis=0)

class Segment(object):
    
    def __init__(self, segments, index):
        
        self.__dict__['_index'] = index
        self.__dict__['_segments'] = segments
    
    @property
    def index(self):
        return self._index
    
    @property
    def segments(self):
        return self._segments
    
    @property
    def indices(self):
        return self.segments._indices[self.index]
    
    @property
    def points(self):
        return self.segments._points[self.segments._indices[self.index]]
    
    @property
    def origin(self):
        if self.segments._origins == 'front':
            return self.segments._points[self.indices][0]
        elif self.segments._origins == 'cop':
            return np.mean(self.segments._points[self.indices],axis=0)
        else:
            raise NotImplementedError()
    
    def __getattr__(self, name):
        """Get name attribute, return None if not explicitely set."""
        
        if name in self.segments._arrays:
            return self.segments._arrays[name][self.index]
        else:
            super().__getattribute__(name)
    
    def __setattr__(self, name, value):
        """Set name attribute to value."""
        
        if name in self.segments._arrays:
            self.segments._arrays[name][self.index] = value
        else:
            super().__setattr__(name, value)