import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.restoration import unwrap_phase
from skimage.filters import gaussian
from scipy.optimize import fmin_powell
from stm.points import Points
from stm.gui import PointsEditor
from stm.peaks import find_local_peaks

class GPA(object):
    
    def __init__(self,image,marker_collection=None,reference_mask=None):
        
        self.reference_mask = reference_mask

        def apply_mask(self, image, point):
        
            if self.mask is None:
                raise RuntimeError('Set a mask')
            
            x, y = np.mgrid[0:self._image.shape[0], 0:self._image.shape[1]]
            
            mask_array = np.ones(self._fft_image.shape)
            
            mask_array = mask_array - self.mask(point.position[0], point.position[1], x, y)
            
            return mask_array * image
    
    def set_reference(self):
        ax=plt.subplot()
        ax.imshow(self.image.T,cmap='gray',interpolation='nearest')
        self.reference_editor = RectangleEditor(ax,self.reference_mask)
        self.reference_editor.create_markers()
        self.reference_mask = self.reference_editor.marker_collection
    
    def get_marker(self,marker_idx):
        return self.marker_collection.markers[marker_idx]
    
    def view(self,method='log-fft',marker_idx=0,ax=None,show_markers=False,scale=10,colorbar=True,cmap='gray',**kwargs):
        
        marker = self.get_marker(marker_idx)
        
        if method == 'log-fft':
            display_image = np.log(1+scale*np.abs(self.fft_image))
            label = 'log(1 + scale * |F|)'
        elif method == 'masked-log-fft':
            display_image = self.apply_mask(np.log(1+scale*np.abs(self.fft_image)),marker)
            label = 'log(1 + scale * |F|)'
        elif method == 'raw-phase':
            display_image = self.raw_phase(marker)
            label = 'Raw phase [rad.]'
        elif method == 'raw-phase-unwrapped':
            display_image = unwrap_phase(self.raw_phase(marker))
            label = 'Raw phase [rad.]'
        elif method == 'reduced-phase':
            display_image = self.reduced_phase(marker)
            label = 'Reduced phase [rad.]'
        elif method == 'reference-phase':
            display_image = self.reference_phase(marker)
            label = 'Reference phase [rad.]'
        elif method == 'residual-reference-phase':
            display_image, optim = self.reciprocal_lattice(marker,return_residual=True)
            label = 'Residual reference phase [rad.]'
        elif method == 'strain-1d':
            display_image = self.strain_1d(marker)*100
            label = 'Strain [%]'
        elif method == 'exx':
            display_image = self.strain()[0,0]*100
            label = 'Strain exx [%]'
        elif method == 'eyy':
            display_image = self.strain()[1,1]*100
            label = 'Strain eyy [%]'
        elif method == 'planar':
            display_image = (self.strain()[0,0]+self.strain()[1,1])/2*100
            label = 'Strain (exx + eyy)/2 [%]'
        elif method == 'exy' or method == 'eyx':
            display_image = self.strain()[1,0]*100
            label = 'Strain exy [%]'
        elif method == 'rotation':
            _, display_image = self.strain(return_rotation=True)/np.pi*180
            label = 'Rotation [deg.]'
        else:
            raise RuntimeError('Method {0} not recognized'.format(method))
        
        if ax is None:
            ax=plt.subplot()
        imshow=ax.imshow(display_image.T,cmap=cmap,**kwargs)
        
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(imshow, cax=cax)
            cbar.set_label(label)
        
        if show_markers:
            self.marker_collection.add_markers_to_ax(ax)
        
        plt.show()
    
    def reference_phase(self,marker):
        limits = np.sort(np.round(self.reference_mask.get_coords()).astype(int),axis=0)
        
        reference_phase = self.raw_phase(marker)[limits[0,0]:limits[1,0],limits[0,1]:limits[1,1]]
        
        return unwrap_phase(reference_phase)
    
    def reciprocal_lattice(self,marker,return_residual=False):
        
        phase_ramp = lambda g: 2.*np.pi*(np.dot(g,[x,y]))
        
        def residual(parameters):
            return np.sum((reference_phase-phase_ramp(parameters[:2])-parameters[2])**2.)
        
        reference_phase = self.reference_phase(marker)
        reference_shape = reference_phase.shape
        reference_phase = reference_phase.ravel()
        
        x,y=np.mgrid[0:reference_shape[0],0:reference_shape[1]]
        x,y = x.ravel(),y.ravel()
        
        initial = np.array(marker.coord).astype(float)/self.fft_image.shape-.5
        initial = np.hstack((initial,[0]))
        
        optim = fmin_powell(residual,initial,disp=False)
        
        if return_residual:
            optim_phase_ramp = np.reshape(phase_ramp(optim[:2])+optim[2],reference_shape)
            residual_phase_ramp = np.reshape(reference_phase,reference_shape)-optim_phase_ramp
            return residual_phase_ramp, optim[:2]
        else:
            return optim[:2]
    
    def raw_phase(self,marker):
        complex_image = np.fft.ifft2(np.fft.ifftshift(self.apply_mask(self.fft_image,marker)))
        return np.angle(complex_image)
    
    def reduced_phase(self,marker):
        x,y = np.mgrid[0:self.fft_image.shape[0],0:self.fft_image.shape[1]]
        x,y = x.ravel(),y.ravel()
        
        g = self.reciprocal_lattice(marker)
        
        phase_ramp =  np.reshape(2.*np.pi*(np.dot(g,[x,y])),self.fft_image.shape)
        raw_phase = self.raw_phase(marker)
        
        return (raw_phase - phase_ramp) % (2*np.pi) - np.pi
        
    def strain_1d(self,marker):
        g = self.reciprocal_lattice(marker)
        
        direction = g/np.linalg.norm(g)
        phi = np.exp(1.j*self.reduced_phase(marker))
        grad_phi = np.dot(np.array(np.gradient(phi)).T,direction).T
        grad_phi = np.imag(grad_phi/phi)/(2.*np.pi)
        
        return -grad_phi/np.linalg.norm(g)
    
    def strain_optimize(self):
        
        strains = []
        directions = []
        for marker in self.marker_collection.markers:
            strains = self.strain_1d(marker)
            g = self.reciprocal_lattice(marker)
            directions.append(g/np.linalg.norm(g))
            
    def strain(self,return_rotation=False):
        
        G = [self.reciprocal_lattice(marker) for marker in self.marker_collection.markers]
        A = np.linalg.pinv(G)
        
        phi = [np.exp(1.j*self.reduced_phase(marker)) for marker in self.marker_collection.markers]
		
        grad_phi = [np.gradient(p,1,1) for p in phi]
        
        grad_phi_x = [np.imag(gp[0]/p).ravel() for p,gp in zip(phi,grad_phi)]
        grad_phi_y = [np.imag(gp[1]/p).ravel() for p,gp in zip(phi,grad_phi)]
		
        grad_phi_x = np.dot(A,np.array(grad_phi_x))
        grad_phi_y = np.dot(A,np.array(grad_phi_y))
        
        strain_tensor = np.empty((2,2)+self.image.shape)
        strain_tensor[0,0] = np.reshape(grad_phi_x[0],self.image.shape)
        strain_tensor[0,1] = np.reshape(grad_phi_x[1],self.image.shape)
        strain_tensor[1,0] = np.reshape(grad_phi_y[0],self.image.shape)
        strain_tensor[1,1] = np.reshape(grad_phi_y[1],self.image.shape)
        strain_tensor = -np.array(strain_tensor)/(2.*np.pi)
        
        if return_rotation:
            rotation = (strain_tensor[0,1]-strain_tensor[1,0])/2.
            return strain_tensor, rotation
        else:
            return strain_tensor
