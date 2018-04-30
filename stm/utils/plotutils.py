import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def discrete_cmap(color_numbers=None, N=10, cmap=plt.cm.tab10):
    
    if color_numbers is None:
        color_numbers = range(N)
    
    
    cmaplist = [cmap(i) for i in np.linspace(0,1,N)]
    cmaplist = [cmaplist[i] for i in color_numbers]
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mcm', cmaplist, len(color_numbers))
    
    return cmap

def add_colorbar(ax, mapable, label=None, loc='right', size='5%', pad=0.05, ticks=None):
    
    if (loc == 'left')|(loc == 'right'):
        orientation = 'vertical'
    else:
        orientation = 'horizontal'
            
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size=size, pad=pad)
    cbar = plt.colorbar(mapable, cax=cax, orientation=orientation, ticks=ticks)
    
    if label is not None:
        cbar.set_label(label)
    
    return cbar

def add_numbering(ax, points):
    for i,point in enumerate(points):
        ax.annotate(i, (point[0],point[1]))
    
def add_segment_patches(ax, segments, colors=None, clim=None, **kwargs):
    
    try:
        segments = [s for i,s in enumerate(segments) if colors.mask[i]==0]
        colors = colors[colors.mask==0]
    except:
        pass
    
    patches = []
    for i,segment in enumerate(segments):
        patches.append(Polygon(segment.points, True))
    
    p = PatchCollection(patches, **kwargs)
    
    ax.add_collection(p)
    
    if not colors is None:
        p.set_array(np.array(colors))
        
        if clim is not None:
            p.set_clim(clim)
    
    return p