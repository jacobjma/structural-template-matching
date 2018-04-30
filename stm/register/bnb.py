import numpy as np
import heapq
from stm.rmsd.kabsch import rmsd_kabsch
from stm.rmsd.qcp import rmsd_qcp
try:
    from stm.rmsd.qcp import rmsd_qcp
except:
    import warnings
    warnings.warn('Unable to import the fast C-version of the QCP-algorithm, using slow version.')
    from stm.rmsd.qcp_slow import rmsd_qcp

def bnb_search(src, dst, rmsd_max=np.inf, min_level=None, max_eval=np.inf, verbose=1):
    
    N=len(dst)
    max_level = N
    
    if min_level is None:
        min_level = max_level
    
    node0 = Node(level=0, permutation=range(N))
    node0.evaluate(src, dst)
    num_eval = 1
    
    if verbose > 0:
        print('Starting search...')
        print('-----')
        
    while max_level >= min_level:
                
        bestnode = Node(level=max_level, permutation=range(N))
        bestnode.evaluate(src, dst)
        upper_bound = bestnode.rmsd
        
        if verbose > 0:
            print('Maximum level: {}, Upper bound: {}'.format(max_level, upper_bound))
        
        num_eval+=1
		
        heap = [node0]
        
        while len(heap) > 0:
            
            node = heapq.heappop(heap)
            
            if verbose > 1:
                #print('RMSD: {:.4f}, level: {}, best RMSD: {:.4f}'.format(node.rmsd, node.level, bestnode.rmsd))
                print('heap size', len(heap), 'upper bound', upper_bound)
            
            if num_eval > max_eval:
                break
            
            if node.rmsd > upper_bound:
                continue
            
            if node.level > 0:
                if node.rmsd / np.sqrt(node.level) > rmsd_max:
                    continue
            
            if node.level == max_level:
                continue
            
            children = node.generate_children()
            
            for child in children:
                
                if child.rmsd is None:
                    child.evaluate(src, dst)
                    num_eval+=1
                    
                if verbose > 1:
                    print('RSSD: {:.4f}, level: {}, best RSSD: {:.4f}'.format(child.rmsd, child.level, bestnode.rmsd))
                
                if child.rmsd < bestnode.rmsd:
                    heapq.heappush(heap, child)
                
                if child.level == max_level:
                    if child.rmsd < upper_bound:
                        if verbose > 0:
                            print('New upper bound: {:.4f} < {:.4f} (after {} evaluations)'.format(child.rmsd, upper_bound, num_eval))
                        
                        bestnode = child
                        upper_bound = child.rmsd
                        
            if len(heap) > 1e6:
                raise RuntimeError('Too many nodes')
        
        max_level -= 1
        
        #if upper_bound < rmsd_max:
        #    break

    rmsd = bestnode.rmsd / np.sqrt(bestnode.level)
    level = bestnode.level
    permutation = bestnode.permutation
    
    if verbose > 0:
        print('-----')
        print('Search complete after {} evaluations'.format(num_eval))
        print('RSSD: {} (RMSD: {})'.format(rmsd * np.sqrt(bestnode.level), rmsd))
        print('level:',level)
        print('final permutation:',permutation)
	
    return rmsd, level, permutation, num_eval

class Node(object):
    
    def __init__(self, level, permutation):
        self._level = level
        self._permutation = permutation
        self._rmsd = None
        self._bound = None
        self._children = None
    
    @property
    def permutation(self):
        return self._permutation
    
    @property
    def rmsd(self):
        return self._rmsd

    @property
    def bound(self):
        return self._bound
        
    @property
    def level(self):
        return self._level
        
    @property
    def children(self):
        return self._children
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._rmsd == other._rmsd
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self._rmsd < other._rmsd
        return NotImplemented
        
    def evaluate(self, src, dst):
        
        if self._level <= 1:
            self._rmsd = 0.
        else:
            #Lp = np.sum(np.linalg.norm(src[:self._level],axis=0)**2)
            #Lt = np.sum(np.linalg.norm(dst[self._permutation[:self._level]],axis=0)**2)
            
            src = src[:self._level] #/ np.sqrt(Lp/Lt)
            dst = dst[self._permutation[:self._level]]
            
            #p = points.coords[:self._level] / np.sqrt(Lp/Lt)
            #t = template.coords[self._permutation[:self._level]]
            
            rmsd = rmsd_qcp(src, dst) * np.sqrt(len(src))
            #rmsd = rmsd_kabsch(scaled_src, permuted_dst)
            
            #print(rmsd)
            
            self._rmsd = rmsd
            
    def generate_children(self):
        
        if self.children is not None:
            return self.children
        
        num_swaps = len(self.permutation) - self.level
        children = []
        for i in range(num_swaps):
            
            permutation = [x for x in self.permutation]
            
            permutation[self.level], permutation[self.level + i] = permutation[self.level + i], permutation[self.level]
            
            child = Node(self.level + 1, permutation)
            
            children += [child]
        
        self._children = children
        
        return children