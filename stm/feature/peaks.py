import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.measurements import center_of_mass, label
from skimage.filters import gaussian
from tqdm import tqdm

from stm.feature.fitting import fit_gaussian, fit_polynomial, BatchFit

def find_local_peaks(image, min_distance, threshold=0, local_threshold=0, 
                    exclude_border=0, exclude_adjacent=False):
    
    """Return peaks in an image as a Points object.
    
    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).
    
    A maximum filter is used for finding local maxima. This operation dilates 
    the original image. After comparison of the dilated and original image, 
    this function returns the coordinates or a mask of the peaks where the 
    dilated image equals the original image.
    
    Parameters
    ----------
    image : ndarray
        Input image.
    min_distance : int
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`).
        To find the maximum number of peaks, use `min_distance=1`.
    threshold : float, optional
        Minimum relative intensity of peaks. By default, the threshold 
        is zero.
    local_threshold : float, optional
        Minimum local relative intensity of peaks. A minimum filter is used 
        for finding the baseline for comparing the local intensity.
        By default, the local threshold is zero.
    exclude_border : int, optional
        If nonzero, `exclude_border` excludes peaks from
        within `exclude_border`-pixels of the border of the image.
    exclude_adjacent : bool, optional
        In case of flat peaks (i.e. multiple adjacent pixels have 
        identical intensities), if true, only the mean pixel position 
        will be returned.
    
    """
    
    image = image.astype(np.float32)
    
    threshold = image.min() + threshold*(image.max()-image.min())
    
    max_filt = maximum_filter(image, min_distance)
    
    is_peak = (image == max_filt)
    
    is_peak[image < threshold] = False
    
    if local_threshold > 0:
        local_threshold = image.min() + local_threshold * (image.max()-image.min())
        min_filt = minimum_filter(image, min_distance)
        is_peak[(max_filt - min_filt) < local_threshold] = False
    
    if exclude_border:
        is_peak[0:exclude_border+1,:] = False
        is_peak[:,0:exclude_border+1] = False
        is_peak[-1:-exclude_border-1:-1,:] = False
        is_peak[:,-1:-exclude_border-1:-1] = False
        
    if exclude_adjacent:
        labels = label(is_peak)
        peaks = center_of_mass(np.ones_like(labels[0]), labels[0], range(1,labels[1]+1))
        peaks = np.array(peaks)
    else:
        peaks = np.array(np.where(is_peak)).T

    return peaks
    
def refine_peaks(image, points, region=3, model='gaussian', progress_bar=False):
    
    """Refine the position intensity extrema to sub-pixel accuracy.

    The positions of intensity extrema are refined by fitting a model function
    to a region close to the pixel positions of local extrema.
    
    Parameters
    ----------
    image : ndarray
        Input image.
    points : Points object
        Points object containing the pixel positions of local extrema
    region : int or ndarray
        
    model : str
        Name of model function fitted to the intensity extrema.
        Should be one of:
            'polynomial'
            'gaussian'
            'elliptical-gaussian'
    progress_bar : bool
        If True, display a progress bar.
    """
    
    if isinstance(region, int):
        region = np.ones((region,)*2,dtype=bool)
    
    if not np.all(np.array(region.shape) % 2):
        RuntimeError()
    
    r = (np.array(region.shape)-1)//2
    
    region = np.array(region).astype(bool)
    
    refined = np.zeros_like(points, dtype=float)
    
    X,Y = np.indices(image.shape)
    
    u,v = np.indices(region.shape)
    u = u[region] - r[0]
    v = v[region] - r[1]
    
    if 'gaussian' in model.lower():
        bf = BatchFit(model=model, N=120, proportiontocut=.4)
    
    for i,p in enumerate(tqdm(points, disable=not progress_bar)):
        x = X[p[0]-r[0]:p[0]+r[0]+1,p[1]-r[1]:p[1]+r[1]+1]
        y = Y[p[0]-r[0]:p[0]+r[0]+1,p[1]-r[1]:p[1]+r[1]+1]
        
        z = image[x,y][region]
        
        if 'gaussian' in model.lower():
            x0, y0 = bf.fit(u, v, z)
        elif model.lower() == 'polynomial':
            x0, y0 = fit_polynomial(u, v, -z)
        else:
            raise NotImplementedError()
        
        refined[i,:] = [x0+p[0],y0+p[1]]
    
    return refined
