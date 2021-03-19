# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import toeplitz

from .propagate_filter import _fir_filter

__all__ = ["convolve_unc"]

def convolve_unc(x1, U1, x2, U2, mode="full"):
    """
    An implementation of the discrete convolution of two signals with uncertainty propagation.
    The code builds on PyDynamic.uncertainty._fir_filter, as convolution and filter application 
    are mathematically nearly identical. However, boundary effects need to be taken into account.

    Parameters
    ----------
    x1 : np.ndarray, (N,)
        first input signal
    U1 : np.ndarray, (N, N)
        full 2D-covariance matrix associated with x1
        if the signal is fully certain, use `U1 = None` (default) to make use of more efficient calculations.
    x2 : np.ndarray, (M,)
        second input signal
    U2 : np.ndarray, (M, M)
        full 2D-covariance matrix associated with x2
        if the signal is fully certain, use `U2 = None` (default) to make use of more efficient calculations.
    mode : str, optional
        numpy.convolve-modes:
        - full:  len(y) == N+M-1 (default)
        - valid: len(y) == max(M, N) - min(M, N) + 1
        - same:  len(y) == max(M, N) (value+covariance are padded with zeros)
        scipy.ndimage.convolve1d-modes:
        - nearest: len(y) == N (value+covariance are padded with by stationary assumption)
        - reflect:  len(y) == N
        - mirror:   len(y) == N

    Returns
    -------
    y : np.ndarray
        convoluted output signal
    Uy : np.ndarray
        full 2D-covariance matrix of y

    References
    ----------
    .. seealso:: :mod:`numpy.convolve`
    .. seealso:: :mod:`scipy.ndimage.convolve1d`
    .. seealso:: :mod:`PyDynamic.uncertainty.propagate_filter._fir_filter`
    """
    
    # if a numpy-mode is chosen, x1 is expected to be the longer signal
    # remeber that pure convolution is commutative
    if len(x1) < len(x2) and mode in ["valid", "full", "same"]:
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
        pad_len = len(x2)-1
        x1_mod = np.pad(x1, (0, pad_len), mode="constant", constant_values=0)
        U1_mod = _pad_covariance(U1, 0, pad_len, mode="zero")

        # apply _fir_filter
        y, Uy = _fir_filter(x=x1_mod, theta=x2, Ux=U1_mod, Utheta=U2, initial_conditions="zero")

        # use output directly
        conv = y
        Uconv = Uy

    elif mode == "same":
        # append (len(x2)-1)//2 to x1
        pad_len = (len(x2)-1)//2
        x1_mod = np.pad(x1, (0, pad_len), mode="constant", constant_values=0)
        U1_mod = _pad_covariance(U1, 0, pad_len, mode="zero")

        # apply _fir_filter
        y, Uy = _fir_filter(x=x1_mod, theta=x2, Ux=U1_mod, Utheta=U2, initial_conditions="zero")

        # remove first (len(x2)-1)//2 entries from output
        conv = y[pad_len:]
        Uconv = Uy[pad_len:, pad_len:]

    elif mode == "nearest":

        # append (len(x2)-1)//2 to x1
        n1 = len(x1)
        n2 = len(x2)
        pad_len = (n2+1)//2
        x1_mod = np.pad(x1, (pad_len, pad_len), mode="edge")
        U1_mod = _pad_covariance(U1, pad_len, pad_len, mode="stationary")

        # apply _fir_filter
        y, Uy = _fir_filter(x=x1_mod, theta=x2, Ux=U1_mod, Utheta=U2, initial_conditions="zero")

        # remove leading and trailing entries from output
        conv = y[n2:n2+n1]
        Uconv = Uy[n2:n2+n1, n2:n2+n1]

    elif mode == "reflect":
        # append (len(x2)-1)//2 to x1
        n1 = len(x1)
        n2 = len(x2)
        pad_len = (n2+1)//2
        x1_mod = np.pad(x1, (pad_len, pad_len), mode="symmetric")
        U1_mod = _pad_covariance(U1, pad_len, pad_len, mode="symmetric")

        # apply _fir_filter
        y, Uy = _fir_filter(x=x1_mod, theta=x2, Ux=U1_mod, Utheta=U2, initial_conditions="zero")

        # remove leading and trailing entries from output
        conv = y[n2:n2+n1]
        Uconv = Uy[n2:n2+n1, n2:n2+n1]

    elif mode == "mirror":
        # append (len(x2)-1)//2 to x1
        n1 = len(x1)
        n2 = len(x2)
        pad_len = (n2+1)//2
        x1_mod = np.pad(x1, (pad_len, pad_len), mode="reflect")
        U1_mod = _pad_covariance(U1, pad_len, pad_len, mode="reflect")

        # apply _fir_filter
        y, Uy = _fir_filter(x=x1_mod, theta=x2, Ux=U1_mod, Utheta=U2, initial_conditions="zero")

        # remove leading and trailing entries from output
        conv = y[n2:n2+n1]
        Uconv = Uy[n2:n2+n1, n2:n2+n1]

    else:
        raise ValueError("convolve_unc: Mode \"{MODE}\" is not supported.".format(MODE=mode))

    return conv, Uconv


def _pad_covariance(U, n_prepend=0, n_append=0, mode="edge"):
    """
        Pad (prepend and append) values to an existing covariance-matrix

        Parameters
        ----------
        U : np.ndarray (N,N)
            covariance matrix to be extended
        n_prepend : int
            how many diagonal elements to extend into the past
        n_append : int
            how many diagonal elements to extend into the future
        mode : string, optional
            - zero: assume zero uncertainty outside
            - stationary: assume that diagonals are stationary before first value and after last value (aaaa|abcd|dddd)
            - symmetric: mirrored along edge , unknown values in off-diagonals are filled by "stationary"-strategy (dcba|abcd|dcba)
            - reflect: mirrored along edge, unknown values in off-diagonals are filled by "stationary"-strategy (dcb|abcd|cba)

        Return
        ------
        U_adjusted : np.ndarray (N+n_prepend+n_append, N+n_prepend+n_append)
            The padded covariance matrix according to mode
        

        Note: 
        The terminology of boundary behavior is not consistent across this function, numpy.pad and scipy.ndimage.convolve1D:
            - zero       <-> numpy constant  <-> scipy constant <-> 0000|abcd|0000
            - stationary <-> numpy edge      <-> scipy nearest  <-> aaaa|abcd|dddd
            - symmetric  <-> numpy symmetric <-> scipy reflect  <-> dcba|abcd|dcba
            - reflect    <-> numpy reflect   <-> scipy mirror   <->  dcb|abcd|cba

    """

    # only append, if U is an array (leave it as None)
    if isinstance(U, np.ndarray):

        if mode == "zero":
            U_adjusted = np.pad(U, ((n_prepend, n_append), (n_prepend, n_append)), mode="constant", constant_values=0.0)

        elif mode in ["stationary", "symmetric", "reflect"]:
            
            # prepend
            c = np.r_[U[:, 0], np.zeros(n_prepend)]
            r = np.r_[U[0, :], np.zeros(n_prepend)]
            U_prepended = toeplitz(c, r)
            U_prepended[n_prepend:, n_prepend:] = U

            # append
            c = np.r_[np.zeros(n_append), U_prepended[:, -1]][::-1]
            r = np.r_[np.zeros(n_append), U_prepended[-1, :]][::-1]
            U_adjusted = toeplitz(c, r)
            U_adjusted[:-n_append, :-n_append] = U_prepended

            # mirror if necessary
            if mode in "symmetric":  # dcba|abcd|dcba
                U_adjusted[:n_prepend,:n_prepend] = U[:n_prepend,:n_prepend][::-1,::-1]
                U_adjusted[-n_append:, -n_append:] = U[-n_append:, -n_append:][::-1,::-1]

            # reflect if necessary
            elif mode in "reflect":  # dcb|abcd|cba
                U_adjusted[:n_prepend+1,:n_prepend+1] = U[:n_prepend+1,:n_prepend+1][::-1,::-1]
                U_adjusted[-n_append-1:, -n_append-1:] = U[-n_append-1:, -n_append-1:][::-1,::-1]

        else:
            raise ValueError("_pad_covariance: Mode \"{MODE}\" is not supported.".format(MODE=mode))
    
    else:
        U_adjusted = None

    return U_adjusted
