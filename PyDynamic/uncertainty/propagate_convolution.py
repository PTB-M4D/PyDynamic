# -*- coding: utf-8 -*-

import numpy as np
from numpy.lib.arraysetops import isin
from .propagate_filter import _fir_filter

__all__ = ["convolve_unc"]

def convolve_unc(x1, U1, x2, U2, mode="full"):
    """
    An implementation of the discrete convolution of two signals with uncertainty propagation.
    The code builds on PyDynamic.uncertainty._fir_filter, as convolution and filter application 
    are mathematically nearly identical. However, boundary effects need to be taken into account.

    Parameters
    ----------
    x1 : np.ndarray
        first input signal
    U1 : np.ndarray
        full 2D-covariance matrix associated with x1
        if the signal is fully certain, use `Ux = None` (default) to make use of more efficient calculations.
    x2 : np.ndarray
        second input signal
    U2 : np.ndarray
        full 2D-covariance matrix associated with x2
        if the signal is fully certain, use `Ux = None` (default) to make use of more efficient calculations.
    mode : str, optional
        full:  (default)
        valid: 
        same: 
    
    Returns
    -------
    y : np.ndarray
        convoluted output signal
    Uy : np.ndarray
        full 2D-covariance matrix of y

    References
    ----------
    .. seealso:: :mod:`numpy.convolve`
    .. seealso:: :mod:`PyDynamic.uncertainty.propagate_filter._fir_filter`
    """
    
    # assume that x1 is the longer signal, otherwise exchange (convolution is commutative)
    if len(x1) < len(x2):
        tmp = x1
        x1 = x2
        x2 = tmp

        tmp = U1
        U1 = U2
        U2 = tmp

    # actual computation
    if mode == "valid":
        # apply _fir_filter directly
        y, Uy = _fir_filter(x=x1, theta=x2, Ux=U1, Utheta=U2, initial_conditions="zero")
        
        # remove first len(x2)-1 entries from output
        conv = y[len(x2)-1:]
        Uconv = Uy[len(x2)-1:, len(x2)-1:]

    elif mode == "full":
        # append len(b)-1 zeros to x1/U1
        x1_mod = np.pad(x1, (0, len(x2)-1), mode="constant", constant_values=0)
        if isinstance(U1, np.ndarray):
            U1_mod = np.pad(U1, ((0, len(x2)-1), (0, len(x2)-1)), mode="constant", constant_values=0)
        else:
            U1_mod = None

        # apply _fir_filter
        y, Uy = _fir_filter(x=x1_mod, theta=x2, Ux=U1_mod, Utheta=U2, initial_conditions="zero")

        # use output directly
        conv = y
        Uconv = Uy

    elif mode == "same":
        # append (len(x2)-1)//2 to x1
        x1_mod = np.pad(x1, (0, (len(x2)-1)//2), mode="constant", constant_values=0)
        if isinstance(U1, np.ndarray):
            U1_mod = np.pad(U1, ((0, (len(x2)-1)//2), (0, (len(x2)-1)//2)), mode="constant", constant_values=0)
        else:
            U1_mod = None
        # apply _fir_filter
        y, Uy = _fir_filter(x=x1_mod, theta=x2, Ux=U1_mod, Utheta=U2, initial_conditions="zero")

        # remove first (len(x2)-1)//2 entries from output
        conv = y[(len(x2)-1)//2:]
        Uconv = Uy[(len(x2)-1)//2:, (len(x2)-1)//2:]

    else:
        raise ValueError("Mode \"{MODE}\" is not supported.".format(MODE=mode))

    return conv, Uconv