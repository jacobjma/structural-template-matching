import numpy as np

def gaussians_image(points, shape=None, image=None, width=1, border=0, cutoff=None):
    
    #points -= np.array([points[:,0].min(), points[:,1].min()])
    #points *= (shape - 2 * border) / np.array([points[:,0].max(), points[:,1].max()])
    #points += border
    
    if cutoff is None:
        cutoff = 4*width
    
    width = 2*width**2
    
    if (image is None)&(shape is not None):
        image = np.zeros(shape)
    elif image is not None:
        shape = image.shape
    else:
        RuntimeError()
    
    x,y = np.mgrid[0:shape[0],0:shape[1]]

    for point in points[:]:
        
        rounded = np.round(point).astype(int)
        indices = (np.abs(x-rounded[0]) < cutoff) & (np.abs(y-rounded[1]) < cutoff)

        image[indices] += np.exp(-((x[indices] - point[0])**2 + (y[indices] - point[1])**2) / width)
    
    return image