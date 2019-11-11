# -*- coding: utf-8 -*-

"""
The :mod:`PyDynamic.uncertainty.propagate_DWT` module implements methods for
the propagation of uncertainties in the application of the discrete wavelet
transform (DWT).
"""

import numpy as np
from propagate_filter import FIRuncFilter


__all__ = ["wavelet_block", "DWT", "DWT_filter_design"]


def wavelet_block(x, Ux, g, h, kind='corr'):
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


def DWT(x, Ux, g, h, depth=-1):
    
    results = []

    level = 0
    x_level = x
    U_level = Ux

    while x_level.size > 1 or level < depth


def DWT_filter_design(length, kind="Haar"):
    pass