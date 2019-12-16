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


__all__ = ["dwt", "wave_dec", "idwt", "wave_rec", "filter_design"]


def dwt(x, Ux, l, h, kind):
    """
    Apply low-pass `l` and high-pass `h` to time-series data `x`.
    The uncertainty is propgated through the transformation by using 
    PyDynamic.uncertainty.FIRuncFilter. and propagate

    Return the subsampled results.

    Parameters
    ----------
        x: np.ndarray
            filter input signal
        Ux: float or np.ndarray
            float:    standard deviation of white noise in x
            1D-array: interpretation depends on kind
        l: np.ndarray
            FIR filter coefficients
            representing a low-pass for decomposition
        h: np.ndarray
            FIR filter coefficients
            representing a high-pass for decomposition
        kind: string
            only meaningfull in combination with isinstance(Ux, numpy.ndarray)
            "diag": point-wise standard uncertainties of non-stationary white noise
            "corr": single sided autocovariance of stationary (colored/corrlated) noise (default)
    
    Returns
    -------
        y_approx: np.ndarray
            subsampled low-pass output signal
        U_approx: np.ndarray
            subsampled low-pass output uncertainty
        y_detail: np.ndarray
            subsampled high-pass output signal
        U_detail: np.ndarray
            subsampled high-pass output uncertainty
    """

    # append signals to compensate for "FIR start"
    pad_len = l.size-1
    x = np.pad(x, (pad_len, pad_len), mode="edge")
    Ux = np.pad(Ux, (pad_len, pad_len), mode="edge")

    # propagate uncertainty through FIR-filter
    y_approx, U_approx = FIRuncFilter(x, Ux, l, Utheta=None, kind=kind)
    y_detail, U_detail = FIRuncFilter(x, Ux, h, Utheta=None, kind=kind)

    # remove "FIR start"-compensation, subsample to half the length
    y_approx = y_approx[pad_len+1::2]
    U_approx = U_approx[pad_len+1::2]
    y_detail = y_detail[pad_len+1::2]
    U_detail = U_detail[pad_len+1::2]

    return y_approx, U_approx, y_detail, U_detail


def idwt(y_approx, U_approx, y_detail, U_detail, l, h, kind):
    """
    Single step of inverse discrete wavelet transform

    Parameters
    ----------
        y_approx: np.ndarray
            low-pass output signal
        U_approx: np.ndarray
            low-pass output uncertainty
        y_detail: np.ndarray
            high-pass output signal
        U_detail: np.ndarray
            high-pass output uncertainty
        l: np.ndarray
            FIR filter coefficients
            representing a low-pass for reconstruction
        h: np.ndarray
            FIR filter coefficients
            representing a high-pass for reconstruction
        kind: string
            only meaningfull in combination with isinstance(Ux, numpy.ndarray)
            "diag": point-wise standard uncertainties of non-stationary white noise
            "corr": single sided autocovariance of stationary (colored/corrlated) noise (default)
    
    Returns
    -------
        x: np.ndarray
            upsampled reconstructed signal
        Ux: np.ndarray
            upsampled uncertainty of reconstructed signal
    """

    # upsample to double the length
    indices = np.arange(1, y_detail.size+1)
    y_approx = np.insert(y_approx, indices, 0)
    U_approx = np.insert(U_approx, indices, 0)
    y_detail = np.insert(y_detail, indices, 0)
    U_detail = np.insert(U_detail, indices, 0)

    # propagate uncertainty through FIR-filter
    x_approx, Ux_approx = FIRuncFilter(y_approx, U_approx, l, Utheta=None, kind=kind)
    x_detail, Ux_detail = FIRuncFilter(y_detail, U_detail, h, Utheta=None, kind=kind)

    # add both parts and remove "FIR start" compensation at the beginning
    ls = l.size - 2
    x = x_detail[ls:] + x_approx[ls:]
    Ux = Ux_detail[ls:] + Ux_approx[ls:]
    return x, Ux


def filter_design(kind):
    """
    Provide low- and highpass filters suitable for discrete wavelet transformation.
    
    Parameters:
    -----------
        kind: string
            filter name, i.e. db4, coif6, gaus9, rbio3.3, ...
            supported families: pywt.families(short=False)
            supported wavelets: pywt.wavelist()

    Returns:
    --------
        ld: np.ndarray
            low-pass filter for decomposition
        hd: np.ndarray
            high-pass filter for decomposition
        lr: np.ndarray
            low-pass filter for reconstruction
        hr: np.ndarray
            high-pass filter for reconstruction
    """

    if kind in pywt.wavelist():
        w = pywt.Wavelet(kind)
        ld = np.array(w.dec_lo)
        hd = np.array(w.dec_hi)
        lr = np.array(w.rec_lo)
        hr = np.array(w.rec_hi)

        return ld, hd, lr, hr

    else:
        raise NotImplementedError("The specified wavelet-kind \"{KIND}\" is not implemented.".format(KIND=kind))


def dwt_max_level(data_length, filter_length):
    n_max = np.floor(np.log2(data_length / (filter_length - 1)))
    return int(n_max)


def wave_dec(x, Ux, lowpass, highpass, n=-1, kind="corr"):
    """
    Multilevel discrete wavelet transformation of time-series x with uncertainty Ux.

    Parameters:
    -----------
        x: np.ndarray
            ...
        Ux: float or np.ndarray
            ...
        lowpass: np.ndarray
            decomposition low-pass for wavelet_block
        highpass: np.ndarray
            decomposition high-pass for wavelet_block
        n: int
            consecutive repetitions of wavelet_block
            user is warned, if it is not possible to reach the specified depth
            ignored if set to -1 (default)
        kind: string
            only meaningfull in combination with isinstance(Ux, numpy.ndarray)
            "diag": point-wise standard uncertainties of non-stationary white noise
            "corr": single sided autocovariance of stationary (colored/corrlated) noise (default)

    Returns:
    --------
        coeffs: list of arrays
            order of arrays within list is:
            [cAn, cDn, cDn-1, ..., cD2, cD1]
        Ucoeffs: list of arrays
            uncertainty of coeffs, same order as coeffs
        original_length: int
            equals to len(x)
            necessary to restore correct length 
    """

    # check if depth is reachable
    max_depth = dwt_max_level(x.size, lowpass.size)
    if n > max_depth:
        raise UserWarning("Will run into trouble, max_depth = {MAX_DEPTH}, but you specified {DEPTH}".format(DEPTH=n, MAX_DEPTH=max_depth))
    elif n == -1:
        n = max_depth

    c_approx = x
    Uc_approx = Ux
    
    original_length = len(x)
    coeffs = []
    Ucoeffs = []

    for level in range(n):
        
        # execute wavelet block
        c_approx, Uc_approx, c_detail, Uc_detail = dwt(c_approx, Uc_approx, lowpass, highpass, kind)

        # save result
        coeffs.insert(0, c_detail)
        Ucoeffs.insert(0, Uc_detail)
        if level + 1 == n:  # save the details when in last level
            coeffs.insert(0, c_approx)
            Ucoeffs.insert(0, Uc_approx)
    
    return coeffs, Ucoeffs, original_length


def wave_rec(coeffs, Ucoeffs, lowpass, highpass, original_length=None, kind="corr"):
    """
    Multilevel discrete wavelet reconstruction of coefficients from levels back into time-series.

    Parameters:
    -----------
        coeffs: list of arrays
            order of arrays within list is:
            [cAn, cDn, cDn-1, ..., cD2, cD1]
            where:
            - cAi: approximation coefficients array from i-th level
            - cDi: detail coefficients array from i-th level
        Ucoeffs: list of arrays
            uncertainty of coeffs, same order as coeffs
        lowpass: np.ndarray
            reconstruction low-pass for wavelet_block
        highpass: np.ndarray
            reconstruction high-pass for wavelet_block
        original_length: optional, int
            necessary to restore correct length of original time-series
    
    Returns
    -------
        x: np.ndarray
            reconstructed signal
        Ux: np.ndarray
            uncertainty of reconstructed signal
    """
    
    # init the approximation coefficients
    c_approx = coeffs[0]
    U_approx = Ucoeffs[0]

    # reconstruction loop
    for c_detail, U_detail in zip(coeffs[1:], Ucoeffs[1:]):
        # crop approximation coefficients if necessary
        lc = len(c_detail)
        c_approx = c_approx[:lc]
        U_approx = U_approx[:lc]

        # execute idwt
        c_approx, U_approx = idwt(c_approx, U_approx, c_detail, U_detail, lowpass, highpass, kind=kind)
    
    # bring to original length (does nothing if original_length == None)
    x = c_approx[:original_length]
    Ux = U_approx[:original_length]

    return x, Ux