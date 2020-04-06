# -*- coding: utf-8 -*-

import numpy as np
from .propagate_filter import FIRuncFilter

__all__ = ["convolve_unc"]

def convolve_unc(x1, U1, x2, U2, mode="full"):
    """
    An implementation of the discrete convolution of two signals with uncertainty propagation.

    The code builds on PyDynamic.uncertainty.FIRuncFilter, as convolution and filter application 
    are mathematically nearly identical. However, boundary effects need to be taken into account.
    """

    # assume that x1 is the longer signal, otherwise exchange
    if len(x1) < len(x2):
        tmp = x1
        x1 = x2
        x2 = tmp

        tmp = U1
        U1 = U2
        U2 = tmp

    # select FIRuncFilter-kind based on U1
    if isinstance(U1, float):
        kind = "float"
    elif isinstance(U1, np.ndarray):
        if len(U1.shape) == 1:
            if U1.size == x1.size:
                kind = "diag"
            else:
                kind = "corr"
        else:
            raise NotImplementedError("Method does currently not support full covariance matrix uncertainty specification of longer signal.")

    # adjust U2 to fit required structure
    if isinstance(U2, float):
        U2 = np.square(U2) * np.eye(len(x2))
    elif isinstance(U2, np.ndarray):
        if len(U2.shape) == 1:
            U2 = np.diag(U2 ** 2)

    # actual computation
    if mode == "valid":
        # apply FIRuncFilter directly
        y, Uy = FIRuncFilter(x1, U1, x2, Utheta=U2, kind=kind)
        
        # remove first len(x2)-1 entries from output
        conv = y[len(x2)-1:]
        Uconv = Uy[len(x2)-1:]

    elif mode == "full":
        # prepend one zero and append len(b)-1 zeros to x1/U1
        x1_mod = np.pad(x1, (1, len(x2)-1), mode="constant", constant_values=0)
        U1_mod = np.pad(U1, (1, len(x2)-1), mode="constant", constant_values=0)

        # apply FIRuncFilter
        y, Uy = FIRuncFilter(x1_mod, U1_mod, x2, Utheta=U2, kind=kind)

        # remove first entry from output
        conv = y[1:]
        Uconv = Uy[1:]

    elif mode == "same":
        # prepend one zero to x1 and append (len(x2)-1)//2
        x1_mod = np.pad(x1, (1, (len(x2)-1)//2), mode="constant", constant_values=0)
        U1_mod = np.pad(U1, (1, (len(x2)-1)//2), mode="constant", constant_values=0)

        # apply FIRuncFilter
        y, Uy = FIRuncFilter(x1_mod, U1_mod, x2, Utheta=U2, kind=kind)

        # remove first (len(x2)+1)//2 entries from output
        conv = y[(len(x2)+1)//2:]
        Uconv = Uy[(len(x2)+1)//2:]

    else:
        raise ValueError("Mode \"{MODE}\" is not supported.".format(MODE=mode))

    return conv, Uconv