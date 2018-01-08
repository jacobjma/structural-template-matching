import numpy as np
from scipy import optimize
from scipy.stats import trim_mean

class BatchFit(object):
    
    def __init__(self, model='elliptical-gaussian', N=100, proportiontocut=0):
        
        self.N = N
        self.proportiontocut = proportiontocut
        
        self._i = 0
        self._model = model
        
        if model == 'elliptical-gaussian':
            self._params = np.zeros((self.N, 7))
            self._predict = np.array([0,0,1,1,1,1,1],dtype=bool)
            self.fit_fun = lambda x, y, z, initial : fit_gaussian(x, y, z, initial, elliptical=True)
        elif model == 'gaussian':
            self._params = np.zeros((self.N, 5))
            self._predict = np.array([0,0,1,1,1],dtype=bool)
            self.fit_fun = lambda x, y, z, initial : fit_gaussian(x, y, z, initial, elliptical=False)
        else:
            raise NotImplementedError()
        
    def fit(self, x, y, z):

        initial = np.empty(self._params.shape[1])
        initial[:] = np.nan
        
        if self._i > self.N:
            if self.proportiontocut == 0:
                initial[self._predict] = np.mean(self._params, axis=0)[self._predict]
            else:
                initial[self._predict] = trim_mean(self._params, self.proportiontocut, axis=0)[self._predict]
        
        params = self.fit_fun(x, y, z, initial)
        
        self._params[self._i % self.N,:][self._predict] = np.array(params)[self._predict]
        
        self._i += 1
        
        return params[0], params[1]
    
    def reset(self):
        self._i = 0

def fit_polynomial(x, y, z, return_params=False):
    
    x=x.ravel()
    y=y.ravel()
    z=z.ravel()
    
    v = np.array([np.ones(len(x)), x, y, x**2, x * y, y**2])
    
    p, residues, rank, singval = np.linalg.lstsq(v.T, z)
    M = [[2 * p[3], p[4]], [p[4], 2 * p[5]]]
    x0, y0 = np.linalg.solve(M, -p[1:3])
    
    if return_params:
        return x0, y0, p
    else:
        return x0, y0

def fit_gaussian(x, y, z, initial, elliptical=True):
    
    if np.isnan(initial[0]):
        initial[0] = (x * z).sum() / z.sum()
    
    if np.isnan(initial[1]):
        initial[1] = (y * z).sum() / z.sum()
    
    if np.isnan(initial[2]):
        initial[2] = z.min()
    
    if np.isnan(initial[3]):
        initial[3] = z.max() - z.min()
    
    if np.isnan(initial[4])|np.isnan(initial[6]):
        _, __, p = fit_polynomial(x, y, z, return_params=True)
    
    x=x.ravel()
    y=y.ravel()
    z=z.ravel()
    
    if elliptical:
        def fun(p):
            x0, y0, z0, A, a, b, c = p
            return A * np.exp(-(a*(x-x0)**2 - 2*b*(x-x0)*(y-y0) + c*(y-y0)**2)) - z
        
        if np.isnan(initial[4]):
            initial[4] = np.abs(p[3])
        
        if np.isnan(initial[5]):
            initial[5] = 0
        
        if np.isnan(initial[6]):
            initial[6] = np.abs(p[5])
        
        size = np.sqrt(len(z))
        
        bounds = [(-size, -size, 0, 0, 0, -size, 0),
                   (size, size, z.max(), np.inf, size, size, size)]
    else:
        if np.isnan(initial[4]):
            initial[4] = (np.abs(p[3]) + np.abs(p[5]))/2
    
        def fun(p): 
            x0, y0, z0, A, a = p
            return z0 + A * np.exp(-a*((x-x0)**2 + (y-y0)**2)) - z
        
        initial = initial[:5]
        
        size = np.sqrt(len(z))
        
        bounds = [(-size, -size, 0, 0, 0),
                   (size, size, z.max(), np.inf)]
    
    ls = optimize.least_squares(fun, initial, bounds=bounds)
    
    return ls.x