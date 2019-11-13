# -*- coding: utf-8 -*-

"""
The :mod:`PyDynamic.uncertainty.propagate_DWT` module implements methods for
the propagation of uncertainties in the application of the discrete wavelet
transform (DWT).
"""

import pywt
import numpy as np
import scipy.signal as scs
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter


__all__ = ["wavelet_block", "DWT", "DWT_filter_design"]


def wavelet_block(x, Ux, g, h, kind):
    """
    Apply low-pass `g` and high-pass `h` to time-series data `x` and propagate
    uncertainty `Ux`. Return the subsampled results.

    To be used as core operation of a wavelet transformation.

    Parameters
    ----------
        x: np.ndarray
            filter input signal
        Ux: float or np.ndarray
            float:    standard deviation of white noise in x
            1D-array: interpretation depends on kind
        g: np.ndarray
            FIR filter coefficients
            representing a low-pass
        h: np.ndarray
            FIR filter coefficients
            representing a high-pass
        kind: string
            only meaningfull in combination with isinstance(sigma_noise, numpy.ndarray)
            "diag": point-wise standard uncertainties of non-stationary white noise
            "corr": single sided autocovariance of stationary (colored/corrlated) noise (default)
    
    Returns
    -------
        y_detail: np.ndarray
            subsampled high-pass output signal
        U_detail: np.ndarray
            subsampled high-pass output uncertainty
        y_approx: np.ndarray
            subsampled low-pass output signal
        U_approx: np.ndarray
            subsampled low-pass output uncertainty
    """

    # propagate uncertainty through FIR-filter
    y_detail, U_detail = FIRuncFilter(x, Ux, h, Utheta=None, kind=kind)
    y_approx, U_approx = FIRuncFilter(x, Ux, g, Utheta=None, kind=kind)

    # subsample to half the length
    y_detail = y_detail[::2] 
    U_detail = U_detail[::2] 
    y_approx = y_approx[::2] 
    U_approx = U_approx[::2] 

    return y_detail, U_detail, y_approx, U_approx


def DWT(x, Ux, g, h, max_depth=-1, kind="corr"):
    """
    Calculate the discrete wavelet transformation of time-series x with uncertainty Ux.
    The uncertainty is propgated through the transformation by using PyDynamic.uncertainty.FIRuncFilter.

    Parameters:
    -----------
        x: np.ndarray
            ...
        Ux: float or np.ndarray
            ...
        g: np.ndarray
            lowpass for wavelet_block
        h: np.ndarray
            high-pass for wavelet_block
        max_depth: int
            maximum consecutive repetitions of wavelet_block
            user is warned, if it is not possible to reach the specified maximum depth
            ignored if set to -1 (default)
        kind: string
            only meaningfull in combination with isinstance(Ux, numpy.ndarray)
            "diag": point-wise standard uncertainties of non-stationary white noise
            "corr": single sided autocovariance of stationary (colored/corrlated) noise (default)

    Returns:
    --------
        tbd, currently just a list of tuples with results of each level.
    """

    y_detail = x
    U_detail = Ux
    
    results = []
    counter = 0

    while True:
        
        # execute wavelet bloc
        y_detail, U_detail, y_approx, U_approx = wavelet_block(y_detail, U_detail, g, h, kind)

        results.append((y_approx, U_approx))


        # if max_depth was specified, check if it is reached
        if (max_depth != -1) and (counter >= max_depth):
            print("reached max depth")
            break
    
        # check if another iteration is possible
        if y_detail.size <= 1:

            # warn user if specified depth is deeper than achieved depth
            if counter < max_depth:
                raise UserWarning("Reached only depth of {COUNTER}, but you specified {MAX_DEPTH}".format(COUNTER=counter, MAX_DEPTH=max_depth))
            break

        counter += 1
    
    return results


def DWT_filter_design(kind, kwargs):
    """
    Provide low- and highpass filters suitable for discrete wavelet transformation.
    
    Parameters:
    -----------
        kind: string
            filter name
        kwargs: dict
            dictionary of keyword arguments for the underlying function scipy.signal.<filtername>
    
    Returns:
    --------
        g: np.ndarray
            low-pass filter
        h: np.ndarray
            high-pass filter
    """

    if kind == "daub":
        g = scs.daub(**kwargs)
        h = scs.qmf(g)

    elif kind == "ricker":
        g = scs.ricker(**kwargs)
        h = scs.qmf(g)

    else:
        raise NotImplementedError("The specified wavelet-kind \"{KIND}\" is not implemented.".format(KIND=kind))

    return g, h