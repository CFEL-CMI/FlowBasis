import numpy as np
#import Tasmanian
from numpy.polynomial.hermite import hermgauss
import jax 
from numpy.polynomial.legendre import leggauss 
import sys 

def Herm1d(npt):
    """One-dimensional Gauss-Hermite quadrature, shifted and scaled

    Args:
    
    - npt : int, Number of quadrature points

    Returns:
    
    - x : array(npt), Quadrature points, points = x / scaling + ref[ind], where x are quadrature abscissas
    - weights : array(npt), Quadrature weights
    """
    x, weights = hermgauss(npt)
    return x, weights

def QuadratureGenerator(nquads, quads, pthr=None, wthr=None, poten=None):
    """Generates a quadrature grid.
   
    Args:
    
    - nquads: list[float], a list specifying number of quadrature points we require per dimension. len(nquads) = #modes in the system. 
    - quads: list[Callable], a list specifying the type of quadrature points we require per mode. len(quads) = #modes in the system.
    - pthr: float, threshold of the potential; quadrature points corresponding a value of the potential that is higher than this get removed. 
    - poten: Callable, a function returning the potential energy evaluated at points. 
    - wthr: a weight threshold; quadrature points corresponding to weights less than this value are eleminated. 
    
    Returns:
    
    points: ndarray; quadrature points. 
    weights: ndarray; corresponding weights.
    """    

    ncoords = len(nquads)
    quadratures = [quad(nquad) for quad, nquad in zip(quads, nquads)]
    points = (quadratures[i][0] for i in range(ncoords))
    weights = (quadratures[i][1] for i in range(ncoords))
    points = np.array(np.meshgrid(*points)).T.reshape(-1, ncoords)
    weights_grid = np.array(np.meshgrid(*weights)).T.reshape(-1, ncoords)
    weights = np.ones((weights_grid.shape[0],))
    
    for i in range(ncoords):
        if nquads[i] != 1: # we're integrating in this dimension
            weights *= weights_grid[:,i]
    
    # remove points with large potential
    if pthr is not None:
        # here I still need to pass the Linear layer to allow for the deletion of quadrature points
        V = poten(points)
        Vmin, Vmax = np.min(V), np.max(V)
        ind = np.where(V-Vmin <= vtol)
        points, weights_grid = points[ind], weights[ind]
        del V
 
    # remove points with small weight
    if wthr is not None:
        ind = np.where(weights > wthr)
        points = points[ind]
        weights = weights[ind]

    return points, weights

if __name__ == '__main__':

    x, weights = QuadratureGenerator([10,10,1], [Herm1d, Herm1d, Herm1d], wthr=1e-2)
    print(x.shape, weights.shape)
