import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from stm.gui import PointsEditor
from stm.feature.peaks import find_local_peaks
from stm.preprocess import normalize

def gaussian_mask(width, ratio=1):
    return lambda x0,y0,x,y: np.exp(-(x0 - x)**2 / (2*(width * ratio)**2) - (y0 - y)**2 / (2*width**2))

def butterworth_mask(width, ratio=1, order=3):
    r = lambda x0,y0,x,y: np.sqrt(((x0 - x) / ratio)**2 + (y0 - y)**2) / width
    return lambda x0,y0,x,y: np.sqrt(1 / (1 + r(x0,y0,x,y)**(2. * order)))

def cosine_mask(width, ratio=1):
    r = lambda x0,y0,x,y: np.sqrt(((x0 - x) / ratio)**2 + (y0 - y)**2) / (width/2)
    return lambda x0,y0,x,y: (1./2+np.cos(r(x0,y0,x,y))/2.)*(r(x0,y0,x,y)<np.pi)

def fft_extent(n, d=1.0):
    if n%2 == 0:
        return [-n/2/(d*n),(n/2-1)/(d*n)]
    else:
        return [-(n-1)/2/(d*n),(n-1)/2/(d*n)]
    
class BraggFilter(object):
    
    """BraggFilter object.

    The BraggFilter object is used filter an image based masking spatial frequencies 
    away from the main spots in a periodic image.
    
    Parameters
    ----------
    image : ndarray
        Input image.
    vectors : Points object
        The reciprocal lattice vectors corresponding to the
    
    """
    
    def __init__(self, image, vectors=None):
        
        self.set_image(image)
        
        self.vectors = vectors
    
    def autoset_vectors(self, min_distance, threshold=0, scale=1, exclusion_radius=None, **kwargs):
        
        """Sets the reciprocal lattice vectors using peak finding. 
        
        The peak finding is performed using stm.find_local_peaks on the normalized 
        transformed power spectrum, P'', given by
            P'' = (P' - min(P')) / (max(P') - min(P'))
        where
            P' = log(1 + s * P)
        and s controls the enhancement of low intensity pixels.
        
        Parameters
        ----------
        min_distance : int
            Minimum number of pixels separating peaks in a region of `2 * min_distance + 1` 
            (i.e. peaks are separated by at least `min_distance`).
            To find the maximum number of peaks, use `min_distance=1`.
        threshold : float, optional
            Minimum relative intensity of peaks. By default, the threshold 
            is zero.
        scale : float
            Constant controlling the enhancement of low intensity pixels. 
        exclusion radius : float
            Radius from zero-frequency component excluded.
        **kwargs :
            Additional keyword arguments for stm.find_local_peaks
        """
        
        pow_spec = np.log(1 + scale*np.abs(self._fft_image))
        
        if exclusion_radius is None:
            exclusion_radius = min_distance/2
        
        if exclusion_radius > 0:
            x,y = np.indices(pow_spec.shape)
            pow_spec[(x-pow_spec.shape[0]/2)**2+(y-pow_spec.shape[1]/2)**2<exclusion_radius**2] = 0
        
        pow_spec = (pow_spec-pow_spec.min())/(pow_spec.max()-pow_spec.min())
        
        pow_spec = gaussian(pow_spec,1)
        
        vectors = find_local_peaks(pow_spec, min_distance, threshold, **kwargs).astype(float)
        
        self.vectors = self.transform_vectors(vectors)
        
    def transform_vectors(self, vectors):
        new_vectors = np.zeros_like(vectors)
    
        extent = fft_extent(self._fft_image.shape[0])
        new_vectors[:,0] = vectors[:,0]*(extent[1] - extent[0])/self._fft_image.shape[0] + extent[0]
        extent = fft_extent(self._fft_image.shape[1])
        new_vectors[:,1] = vectors[:,1]*(extent[0] - extent[1])/self._fft_image.shape[1] + extent[1]
        return new_vectors
    
    def untransform_vectors(self, vectors):
        new_vectors = np.zeros_like(vectors)
    
        extent = fft_extent(self._fft_image.shape[0])
        new_vectors[:,0] = (vectors[:,0]-extent[0])*self._fft_image.shape[0]/(extent[1] - extent[0])
        extent = fft_extent(self._fft_image.shape[1])
        new_vectors[:,1] = (vectors[:,1]-extent[1])*self._fft_image.shape[1]/(extent[0] - extent[1])
        return new_vectors
        
    def display_vectors(self, ax=None, scale=1, facecolors='none', edgecolors='r', **kwargs):
        
        if ax is None:
            ax=plt.subplot()
        
        pow_spec = np.log(1 + scale*np.abs(self._fft_image)).T
        
        ax.imshow(pow_spec, cmap='gray', interpolation='nearest', 
                    extent=fft_extent(self._fft_image.shape[0])+fft_extent(self._fft_image.shape[1]))
        
        if self.vectors is not None:
            ax.scatter(self.vectors[:,0], self.vectors[:,1], facecolors=facecolors, 
                        edgecolors=edgecolors, **kwargs)
        
        return ax
    
    def display_mask(self, ax, mask):
        
        if ax is None:
            ax=plt.subplot()
        
        shape = self._fft_image.shape
        
        mask_array = self._get_mask(shape, mask)
        
        ax.imshow(mask_array.T, extent=fft_extent(shape[0]) + fft_extent(shape[1]))
        
        return ax
    
    def set_vectors(self, vectors):
        self.vectors = vectors
    
    def edit_vectors(self, scale=10, **kwargs):
        """Edit reciprocal lattice vectors.
        
        This method uses the Points editor to manually set the reciprocal lattice vectors.
        
        Parameters
        ----------
        scale : float
            Constant controlling the enhancement of low intensity pixels. This only modifies
            the visual appearance. 
        """
        
        ax=plt.subplot()
        
        pow_spec = np.log(1 + scale*np.abs(self._fft_image)).T
        
        ax.imshow(pow_spec, cmap='gray', interpolation='nearest', 
                extent=fft_extent(pow_spec.shape[0])+fft_extent(pow_spec.shape[1]))
        
        if self.vectors is None:
            self.vectors = np.zeros((0,2))

        self.pe = PointsEditor(ax, self.vectors)
        
        self.pe.edit(close_callback = self.set_vectors, **kwargs)
    
    def set_image(self, image):
        """Set the image to be filtered.
        
        Parameters
        ----------
        image : ndarray
            Input image.
        """
    
        self._image = (image-image.min())/(image.max()-image.min())
        self._fft_image = np.fft.fftshift(np.fft.fft2(image))
    
    def _get_mask_array(self, point, shape, mask):
        
        x, y = np.mgrid[0:shape[0], 0:shape[1]]
        
        mask_array = np.zeros(shape)
        
        mask_array = mask_array + mask(point[0], point[1], x, y)
        
        return mask_array
    
    def _get_mask(self, shape, mask):
        mask_array = np.zeros(shape)
        
        points = self.untransform_vectors(self.vectors)
        
        for point in points:
            mask_array += self._get_mask_array(point, shape, mask)
        
        return mask_array
    
    def apply_filter(self, mask, image=None, return_mask=False):
        """Apply Bragg filter to image.
        
        Parameters
        ----------
        mask : callable
            The function defining the Bragg filter mask(s). 
            Must be in the form mask(x0, y0, x, y). The mask center 
            image indices, (x0,y0), are floats, and the image indices 
            x and y are ndarrays.
        image : ndarray
            Input image. Defaults to the previously set image.
        return_mask : bool
            If True, also return the mask array.
            
        Returns
        ----------
        filter image : ndarray
            The Bragg filtered image.
        filter image : ndarray, optional
            The mask array multiplied with the fourier transformed image.
        """
        
        if image is None:
            fft_image = self._fft_image.copy()
        else:
            fft_image = np.fft.fftshift(np.fft.fft2(image))
        
        mask_array = self._get_mask(fft_image.shape, mask)
        
        filtered_image = np.fft.fft2(np.fft.fftshift(fft_image * mask_array)).real
        
        filtered_image = normalize(filtered_image)
        
        if return_mask:
            return filtered_image, mask_array
        else:
            return filtered_image
