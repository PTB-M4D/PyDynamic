"""This module assists in uncertainty propagation for the convolution operation

The convolution operation is a common operation in signal and data
processing. Convolving signals is mathematically similar to a filter
application. 

This module contains the following function:

* :func:`convolve_unc`: Convolution with uncertainty propagation based on FIR-filter
"""

__all__ = ["convolve_unc"]

import numpy as np

from .propagate_filter import _fir_filter
from ..misc.tools import is_vector


def convolve_unc(x1, U1, x2, U2, mode="full"):
    """Discrete convolution of two signals with uncertainty propagation

    This function supports the convolution modes of :func:`numpy.convolve` and
    :func:`scipy.ndimage.convolve1d`.

    .. note::
        The option to provide the uncertainties as 1D-arrays of standard uncertainties
        is given for convenience only. It does not result in any performance benefits,
        as they are internally just converted into a diagonal covariance matrix.
        Moreover, the output will always be a full covariance matrix (and will almost
        always have off-diagonal entries in practical scenarios).

    Parameters
    ----------
    x1 : np.ndarray, (N,)
        first input signal
    U1 : np.ndarray, (N,) or (N, N)
        - 1D-array: standard uncertainties associated with x1 (corresponding to uncorrelated entries of x1)
        - 2D-array: full 2D-covariance matrix associated with x1
        - None: corresponds to a fully certain signal x1, results in more efficient calculation (compared to using np.zeros(...))
    x2 : np.ndarray, (M,)
        second input signal
    U2 : np.ndarray, (M,) or (M, M)
        - 1D-array: standard uncertainties associated with x2 (corresponding to uncorrelated entries of x2)
        - 2D-array: full 2D-covariance matrix associated with x2
        - None: corresponds to a fully certain signal x2, results in more efficient calculation (compared to using np.zeros(...))
    mode : str, optional
        :func:`numpy.convolve`-modes:

        - full:  len(y) == N+M-1 (default)
        - valid: len(y) == max(M, N) - min(M, N) + 1
        - same:  len(y) == max(M, N) (value+covariance are padded with zeros)

        :func:`scipy.ndimage.convolve1d`-modes:

        - nearest: len(y) == N (value+covariance are padded with by stationary assumption)
        - reflect:  len(y) == N
        - mirror:   len(y) == N

    Returns
    -------
    conv : np.ndarray
        convoluted output signal
    Uconv : np.ndarray
        full 2D-covariance matrix of y

    References
    ----------
    .. seealso::
        :func:`numpy.convolve`
        :func:`scipy.ndimage.convolve1d`
    """

    # if a numpy-mode is chosen, x1 is expected to be the longer signal
    # remember that pure convolution is commutative
    if len(x1) < len(x2) and mode in ["valid", "full", "same"]:
        x1, x2 = x2, x1
        U1, U2 = U2, U1

    # convert 1d array of standard uncertainties to covariance matrix (if necessary)
    U1, U2 = _ensure_cov_matrix(U1), _ensure_cov_matrix(U2)

    # actual computation
    if mode == "valid":
        # apply _fir_filter directly
        y, Uy = _fir_filter(x=x1, theta=x2, Ux=U1, Utheta=U2, initial_conditions="zero")

        # compensate boundary adjustments from _fir_filter
        conv = y[len(x2) - 1 :]
        Uconv = Uy[len(x2) - 1 :, len(x2) - 1 :]

    elif mode == "full":
        # append zeros to adapt to _fir_filter mechanism
        pad_len = len(x2) - 1
        x1_mod = np.pad(x1, (0, pad_len), mode="constant", constant_values=0)
        if isinstance(U1, np.ndarray):
            U1_mod = np.pad(
                U1, ((0, pad_len), (0, pad_len)), mode="constant", constant_values=0
            )
        else:
            U1_mod = None

        # apply _fir_filter
        y, Uy = _fir_filter(
            x=x1_mod, theta=x2, Ux=U1_mod, Utheta=U2, initial_conditions="zero"
        )

        # use output directly
        conv = y
        Uconv = Uy

    elif mode == "same":
        # append zeros to adapt to _fir_filter mechanism
        pad_len = (len(x2) - 1) // 2
        x1_mod = np.pad(x1, (0, pad_len), mode="constant", constant_values=0)
        if isinstance(U1, np.ndarray):
            U1_mod = np.pad(
                U1, ((0, pad_len), (0, pad_len)), mode="constant", constant_values=0
            )
        else:
            U1_mod = None

        # apply _fir_filter
        y, Uy = _fir_filter(
            x=x1_mod, theta=x2, Ux=U1_mod, Utheta=U2, initial_conditions="zero"
        )

        # compensate boundary adjustments from _fir_filter
        conv = y[pad_len:]
        Uconv = Uy[pad_len:, pad_len:]

    elif mode in ["nearest", "reflect", "mirror"]:

        # scipy.ndimage.convolve1d and numpy.pad use different (but overlapping) terminology
        mode_translation = {
            "nearest": "edge",
            "reflect": "symmetric",
            "mirror": "reflect",
        }
        pad_mode = mode_translation[mode]

        # prepend and append to x1 and U1 to get requested boundary effect
        n1 = len(x1)
        n2 = len(x2)
        pad_len = (n2 + 1) // 2
        x1_mod = np.pad(x1, (pad_len, pad_len), mode=pad_mode)
        # we assume that U1 is an array or None
        if isinstance(U1, np.ndarray):
            U1_mod = np.pad(U1, ((pad_len, pad_len), (pad_len, pad_len)), mode=pad_mode)
        else:
            U1_mod = None

        # apply _fir_filter
        y, Uy = _fir_filter(
            x=x1_mod, theta=x2, Ux=U1_mod, Utheta=U2, initial_conditions="zero"
        )

        # compensate boundary adjustments from _fir_filter
        conv = y[n2 : n2 + n1]
        Uconv = Uy[n2 : n2 + n1, n2 : n2 + n1]

    else:
        raise ValueError(f'convolve_unc: Mode "{mode}" is not supported.')

    return conv, Uconv


def _ensure_cov_matrix(unc_array):
    """
    Converts 1D-arrays of standard uncertainties into the corresponding
    (diagonal) covariance matrix by *square*+diag.

    Does not modify inputs which are 2D-arrays or None.
    """

    # squeeze() ensures proper execution on arrays with only one dimension of non-zero length
    if unc_array is not None and is_vector(unc_array):
        unc_array = np.diag(np.square(unc_array))

    return unc_array
