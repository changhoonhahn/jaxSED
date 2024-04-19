'''

some utility functions 

'''
import os
import jax.numpy as np
import jax.scipy as sp
from astropy.io import fits 


# --- constants ---- 
def Lsun(): 
    return 3.846e33  # erg/s


def parsec(): 
    return 3.085677581467192e18  # in cm


def to_cgs(): # at 10pc 
    lsun = Lsun()
    pc = parsec()
    return lsun/(4.0 * np.pi * (10 * pc)**2) 


def c_light(): # AA/s
    return 2.998e18


def jansky_cgs(): 
    return 1e-23


def tlookback_bin_edges(tage): 
    ''' hardcoded log-spaced lookback time bin edges. Bins have 0.1 log10(Gyr)
    widths. See `nb/tlookback_binning.ipynb` for comparison of linear and
    log-spaced binning. With log-space we reproduce spectra constructed from
    high time resolution SFHs more accurately with fewer stellar populations.
    '''
    bin_edges = np.zeros(43)
    bin_edges[1:-1] = 10**(6.05 + 0.1 * np.arange(41) - 9.)
    bin_edges[-1] = 13.8
    if tage is None: 
        return bin_edges
    else: 
        return np.concatenate([bin_edges[bin_edges < tage], [tage]])


def centers2edges(centers):
    """Convert bin centers to bin edges, guessing at what you probably meant
    Args:
        centers (array): bin centers,
    Returns:
        array: bin edges, lenth = len(centers) + 1
    """
    centers = np.asarray(centers)
    edges = np.zeros(len(centers)+1)
    #- Interior edges are just points half way between bin centers
    #edges[1:-1] = (centers[0:-1] + centers[1:]) / 2.0
    edges = edges.at[1:-1].set((centers[0:-1] + centers[1:]) / 2.0)

    #- edge edges are extrapolation of interior bin sizes
    #edges[0] = centers[0] - (centers[1]-edges[1])
    #edges[-1] = centers[-1] + (centers[-1]-edges[-2])
    edges = edges.at[0].set(centers[0] - (centers[1]-edges[1]))
    edges = edges.at[-1].set(centers[-1] + (centers[-1]-edges[-2]))

    return edges



def trapz_rebin(x, y, xnew=None, edges=None):
    ''' jax-friendly version of trapezoidal rebinning
    '''
    if edges is None:
        edges = centers2edges(xnew)
    else:
        edges = np.asarray(edges)

    if edges[0] < x[0] or x[-1] < edges[-1]:
        raise ValueError('edges must be within input x range')

    nbin = len(edges) - 1
    
    results = np.zeros(nbin) 

    for i in range(nbin):  
        j = np.arange(len(x))[x > edges[i]][0]

        yedge = y[j-1] + (edges[i]-x[j-1]) * (y[j]-y[j-1]) / (x[j]-x[j-1])

        #- Is this sample inside this bin?
        if x[j] < edges[i+1]:
            area = 0.5 * (y[j] + yedge) * (x[j] - edges[i])
            results = results.at[i].set(results[i] + area)

            #- Continue with interior bins
            while x[j+1] < edges[i+1]:
                j += 1
                area = 0.5 * (y[j] + y[j-1]) * (x[j] - x[j-1])
                results = results.at[i].set(results[i] + area)

            #- Next sample will be outside this bin; handle upper edge
            yedge = y[j] + (edges[i+1]-x[j]) * (y[j+1]-y[j]) / (x[j+1]-x[j])
            area = 0.5 * (yedge + y[j]) * (edges[i+1] - x[j])
            results = results.at[i].set(results[i] + area)

        #- Otherwise the samples span over this bin
        else:
            ylo = y[j] + (edges[i]-x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            yhi = y[j] + (edges[i+1]-x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            area = 0.5 * (ylo+yhi) * (edges[i+1]-edges[i])
            results = results.at[i].set(results[i] + area)

    for i in range(nbin):
        results = results.at[i].set(results[i] /(edges[i+1] - edges[i]))

    return results


# scipy.ndimage.gaussian_filter1d implementation for jax from 
# https://github.com/renecotyfanboy/jax/blob/gaussian-blur/jax/_src/scipy/ndimage.py

def gaussian_filter1d(x, sigma, axis=-1, order=0,
                      truncate=4.0, 
                      radius=0,
                      mode='constant',
                      cval=0.0,
                      precision=None):
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)

    if mode != 'constant' or cval != 0.:
        raise NotImplementedError('Other modes than "constant" with 0. fill value are not'
                                  'supported yet.')

    if radius > 0.:
        lw = radius
    if lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')

    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]

    # Be careful that modes in signal.convolve refer to the 'same' 'full' 'valid' modes
    # while in gaussian_filter1d refers to the way the padding is done 'constant' 'reflect' etc.
    # We should change the convolve backend for further features
    return np.apply_along_axis(sp.signal.convolve, axis, x, weights,
                                mode='same',
                                method='auto',
                                precision=precision)

def _gaussian_kernel1d(sigma,
                       order,
                       radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q = q.at[0].set(1)
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x

