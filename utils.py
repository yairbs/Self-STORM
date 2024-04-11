import numpy as np

def create_gaussian_filter(sigma=1, mu=0, size=30, mask_thresh=1e-06, sf=1):
    min_lim = -(size-1)/2
    max_lim = (size-1)/2
    x, y = np.meshgrid(np.linspace(min_lim,max_lim,size), np.linspace(min_lim,max_lim,size))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    mask = g < mask_thresh
    g[mask] = 0
    g = g/(np.linalg.norm(g.reshape(-1)))
    g = g*sf
    return g