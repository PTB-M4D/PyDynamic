# -*- coding: utf-8 -*-

"""
The :mod:`PyDynamic.uncertainty.propagate_DWT` module implements methods for
the propagation of uncertainties in the application of the discrete wavelet
transform (DWT).

This modules contains the following functions:

* :func:`dwt`: single level DWT
* :func:`wave_dec`: wavelet decomposition / multi level DWT
* :func:`wave_dec_realtime"`: multi level DWT 
* :func:`idwt`: single level inverse DWT
* :func:`wave_rec`: wavelet reconstruction / multi level inverse DWT
* :func:`filter_desig`: provide common wavelet filters (via :py:mod:`PyWavelets`)

"""

import numpy as np
import pywt

from .propagate_filter import IIRuncFilter, get_initial_state

__all__ = ["dwt", "wave_dec", "wave_dec_realtime",  "idwt", "wave_rec", "filter_design"]


def dwt(x, Ux, l, h, kind, states=None, realtime=False, subsample_start=1):
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
        states: dictionary of internal high/lowpass-filter states
            allows to continue at the last used internal state from previous call
        realtime: Boolean
            for realtime applications, no signal padding has to be done before decomposition
    
    Returns
    -------
        c_approx: np.ndarray
            subsampled low-pass output signal
        U_approx: np.ndarray
            subsampled low-pass output uncertainty
        c_detail: np.ndarray
            subsampled high-pass output signal
        U_detail: np.ndarray
            subsampled high-pass output uncertainty
        states: dictionary of internal high/lowpass-filter states
            allows to continue at the last used internal state in next call
    """

    # prolongate signals if no realtime is needed
    pad_len = 0
    if not realtime:
        pad_len = l.size-1
        x = np.pad(x, (0, pad_len), mode="edge")
        Ux = np.pad(Ux, (0, pad_len), mode="edge")

    # init states if not given
    if not states:
        states = {}
        states["low"] = get_initial_state(l, [1.0], Uab=None, x0=x[0], U0=Ux[0])
        states["high"] = get_initial_state(h, [1.0], Uab=None, x0=x[0], U0=Ux[0])

    # propagate uncertainty through FIR-filter
    c_approx, U_approx, states["low"] = IIRuncFilter(x, Ux, l, [1.0], Uab=None, kind=kind, state=states["low"])
    c_detail, U_detail, states["high"] = IIRuncFilter(x, Ux, h, [1.0], Uab=None, kind=kind, state=states["high"])

    # subsample to half the length
    c_approx = c_approx[subsample_start::2]
    U_approx = U_approx[subsample_start::2]
    c_detail = c_detail[subsample_start::2]
    U_detail = U_detail[subsample_start::2]

    return c_approx, U_approx, c_detail, U_detail, states


def idwt(c_approx, U_approx, c_detail, U_detail, l, h, kind, states=None, realtime=False):
    """
    Single step of inverse discrete wavelet transform

    Parameters
    ----------
        c_approx: np.ndarray
            low-pass output signal
        U_approx: np.ndarray
            low-pass output uncertainty
        c_detail: np.ndarray
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
        states: dictionary of internal high/lowpass-filter states
            allows to continue at the last used internal state from previous call
        realtime: Boolean
            for realtime applications, no signal padding has to be undone after reconstruction
    
    Returns
    -------
        x: np.ndarray
            upsampled reconstructed signal
        Ux: np.ndarray
            upsampled uncertainty of reconstructed signal
        states: dictionary of internal high/lowpass-filter states
            allows to continue at the last used internal state in next call
    """

    # upsample to double the length
    indices = np.arange(1, c_detail.size+1)
    c_approx = np.insert(c_approx, indices, 0)
    U_approx = np.insert(U_approx / np.sqrt(2), indices, 0)  # why is this correction necessary?
    c_detail = np.insert(c_detail, indices, 0)
    U_detail = np.insert(U_detail / np.sqrt(2), indices, 0)  # why is this correction necessary?

    # init states if not given
    if not states:
        states = {}
        states["low"] = get_initial_state(l, [1.0], Uab=None, x0=0, U0=0)   # the value before the first entry is a zero,
        states["high"] = get_initial_state(h, [1.0], Uab=None, x0=0, U0=0)  # if the upsampling would continue into the past

    # propagate uncertainty through FIR-filter
    x_approx, Ux_approx, states["low"] = IIRuncFilter(c_approx, U_approx, l, [1.0], Uab=None, kind=kind, state=states["low"])
    x_detail, Ux_detail, states["high"] = IIRuncFilter(c_detail, U_detail, h, [1.0], Uab=None, kind=kind, state=states["high"])

    # add both parts
    if realtime:
        x = x_detail + x_approx
        Ux = Ux_detail + Ux_approx
    else:
        # remove prolongation if not realtime
        ls = l.size - 2
        x = x_detail[ls:] + x_approx[ls:]
        Ux = Ux_detail[ls:] + Ux_approx[ls:]

    return x, Ux, states


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


def wave_dec(x, Ux, lowpass, highpass, n=-1, kind="diag"):
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
        c_approx, Uc_approx, c_detail, Uc_detail, states = dwt(c_approx, Uc_approx, lowpass, highpass, kind)

        # save result
        coeffs.insert(0, c_detail)
        Ucoeffs.insert(0, Uc_detail)
        if level + 1 == n:  # save the details when in last level
            coeffs.insert(0, c_approx)
            Ucoeffs.insert(0, Uc_approx)
    
    return coeffs, Ucoeffs, original_length


def wave_dec_realtime(x, Ux, lowpass, highpass, n=1, kind="diag", level_states=None):

    if level_states == None:
        level_states = {level: None for level in range(n)}
        level_states["counter"] = 0

    c_approx = x
    Uc_approx = Ux
    
    original_length = len(x)
    coeffs = []
    Ucoeffs = []
    i0 = level_states["counter"]

    for level in range(n):
        # check, where subsampling needs to start
        # (to remain consistency over multiple calls of wave_dec with argument-lengths not equal to 2^n)
        i_n = i0 // 2**level
        subsample_start = (i_n+1)%2

        # execute wavelet block
        if len(c_approx) > 0:
            c_approx, Uc_approx, c_detail, Uc_detail, level_states[level] = dwt(c_approx, Uc_approx, lowpass, highpass, kind, realtime=True, states=level_states[level], subsample_start=subsample_start)
        else:
            c_approx = np.empty(0)
            Uc_approx = np.empty(0)
            c_detail = np.empty(0)
            Uc_detail = np.empty(0)

        # save result
        coeffs.insert(0, c_detail)
        Ucoeffs.insert(0, Uc_detail)
        if level + 1 == n:  # save the details when in last level
            coeffs.insert(0, c_approx)
            Ucoeffs.insert(0, Uc_approx)
    
    # update total counter modulo 2^n
    level_states["counter"] = (level_states["counter"] + len(x)) % 2**n

    return coeffs, Ucoeffs, original_length, level_states


def wave_rec(coeffs, Ucoeffs, lowpass, highpass, original_length=None, kind="diag"):
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
        c_approx, U_approx, states = idwt(c_approx, U_approx, c_detail, U_detail, lowpass, highpass, kind=kind)
    
    # bring to original length (does nothing if original_length == None)
    x = c_approx[:original_length]
    Ux = U_approx[:original_length]

    return x, Ux