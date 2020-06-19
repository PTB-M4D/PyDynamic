# -*- coding: utf-8 -*-

"""
The :mod:`PyDynamic.uncertainty.propagate_DWT` module implements methods for
the propagation of uncertainties in the application of the discrete wavelet
transform (DWT).

This modules contains the following functions:

* :func:`dwt`: single level DWT
* :func:`wave_dec`: wavelet decomposition / multi level DWT
* :func:`wave_dec_realtime"`: multi level DWT 
* :func:`inv_dwt`: single level inverse DWT
* :func:`wave_rec`: wavelet reconstruction / multi level inverse DWT
* :func:`filter_design`: provide common wavelet filters (via :py:mod:`PyWavelets`)
* :func:`dwt_max_level`: return the maximum achievable DWT level

"""

import numpy as np
import pywt

from .propagate_filter import IIRuncFilter, IIR_get_initial_state

__all__ = [
    "dwt",
    "wave_dec",
    "wave_dec_realtime",
    "inv_dwt",
    "wave_rec",
    "filter_design",
    "dwt_max_level",
]


def dwt(x, Ux, lowpass, highpass, states=None, realtime=False, subsample_start=1):
    """
    Apply low-pass `lowpass` and high-pass `highpass` to time-series data `x`.
    The uncertainty is propagated through the transformation by using 
    :func:`PyDynamic.uncertainty.IIRuncFilter`.

    Return the subsampled results.

    Parameters
    ----------
        x : np.ndarray
            filter input signal
        Ux : float or np.ndarray
            float:    standard deviation of white noise in x
            1D-array: point-wise standard uncertainties of non-stationary white noise 
        lowpass : np.ndarray
            FIR filter coefficients
            representing a low-pass for decomposition
        highpass : np.ndarray
            FIR filter coefficients
            representing a high-pass for decomposition
        states : dictionary of internal high/lowpass-filter states, optional (default: None)
            allows to continue at the last used internal state from previous call
        realtime : Boolean, optional (default: False)
            for realtime applications, no signal padding has to be done before decomposition
        subsample_start : int, optional (default: 1)
            At which position the subsampling should start, typically 1 (default) or 0. 
            You should be happy with the default. We only need this to realize :func:`wave_dec_realtime`. 
    
    Returns
    -------
        c_approx  : np.ndarray
            subsampled low-pass output signal
        U_approx : np.ndarray
            subsampled low-pass output uncertainty
        c_detail : np.ndarray
            subsampled high-pass output signal
        U_detail : np.ndarray
            subsampled high-pass output uncertainty
        states : dictionary of internal high/lowpass-filter states
            allows to continue at the last used internal state in next call
    """

    # prolongate signals if no realtime is needed
    if not realtime:
        pad_len = lowpass.size - 1
        x = np.pad(x, (0, pad_len), mode="edge")
        Ux = np.pad(Ux, (0, pad_len), mode="edge")

    # init states if not given
    if not states:
        states = {
            "low": IIR_get_initial_state(lowpass, [1.0], Uab=None, x0=x[0], U0=Ux[0]),
            "high": IIR_get_initial_state(highpass, [1.0], Uab=None, x0=x[0], U0=Ux[0]),
        }

    # propagate uncertainty through FIR-filter
    c_approx, U_approx, states["low"] = IIRuncFilter(
        x, Ux, lowpass, [1.0], Uab=None, kind="diag", state=states["low"]
    )
    c_detail, U_detail, states["high"] = IIRuncFilter(
        x, Ux, highpass, [1.0], Uab=None, kind="diag", state=states["high"]
    )

    # subsample to half the length
    c_approx = c_approx[subsample_start::2]
    U_approx = U_approx[subsample_start::2]
    c_detail = c_detail[subsample_start::2]
    U_detail = U_detail[subsample_start::2]

    return c_approx, U_approx, c_detail, U_detail, states


def inv_dwt(
    c_approx,
    U_approx,
    c_detail,
    U_detail,
    lowpass,
    highpass,
    states=None,
    realtime=False,
):
    """
    Single step of inverse discrete wavelet transform

    Parameters
    ----------
        c_approx : np.ndarray
            low-pass output signal
        U_approx : np.ndarray
            low-pass output uncertainty
        c_detail : np.ndarray
            high-pass output signal
        U_detail : np.ndarray
            high-pass output uncertainty
        lowpass : np.ndarray
            FIR filter coefficients
            representing a low-pass for reconstruction
        highpass : np.ndarray
            FIR filter coefficients
            representing a high-pass for reconstruction
        states : dictionary of internal high/lowpass-filter states, optional (default: None)
            allows to continue at the last used internal state from previous call
        realtime : Boolean, optional (default: False)
            for realtime applications, no signal padding has to be undone after reconstruction
    
    Returns
    -------
        x : np.ndarray
            upsampled reconstructed signal
        Ux : np.ndarray
            upsampled uncertainty of reconstructed signal
        states : dictionary of internal high/lowpass-filter states
            allows to continue at the last used internal state in next call
    """

    # upsample to double the length
    indices = np.arange(1, c_detail.size + 1)
    c_approx = np.insert(c_approx, indices, 0)
    U_approx = np.insert(
        U_approx / np.sqrt(2), indices, 0
    )  # why is this correction necessary?
    c_detail = np.insert(c_detail, indices, 0)
    U_detail = np.insert(
        U_detail / np.sqrt(2), indices, 0
    )  # why is this correction necessary?

    # init states if not given
    if not states:
        states = {
            "low": IIR_get_initial_state(
                lowpass, [1.0], Uab=None, x0=0, U0=0
            ),  # the value before the first entry is a zero, if the upsampling would continue into the past
            "high": IIR_get_initial_state(highpass, [1.0], Uab=None, x0=0, U0=0),
        }

    # propagate uncertainty through FIR-filter
    x_approx, Ux_approx, states["low"] = IIRuncFilter(
        c_approx, U_approx, lowpass, [1.0], Uab=None, kind="diag", state=states["low"]
    )
    x_detail, Ux_detail, states["high"] = IIRuncFilter(
        c_detail, U_detail, highpass, [1.0], Uab=None, kind="diag", state=states["high"]
    )

    # add both parts
    if realtime:
        x = x_detail + x_approx
        Ux = Ux_detail + Ux_approx
    else:
        # remove prolongation if not realtime
        ls = lowpass.size - 2
        x = x_detail[ls:] + x_approx[ls:]
        Ux = Ux_detail[ls:] + Ux_approx[ls:]

    return x, Ux, states


def filter_design(kind):
    """
    Provide low- and highpass filters suitable for discrete wavelet transformation.
    This wraps :py:mod:`PyWavelets`.
    
    Parameters:
    -----------
        kind : string
            filter name, i.e. db4, coif6, gaus9, rbio3.3, ...
            supported families: :func:`pywt.families`
            supported wavelets: :func:`pywt.wavelist`

    Returns:
    --------
        ld : np.ndarray
            low-pass filter for decomposition
        hd : np.ndarray
            high-pass filter for decomposition
        lr : np.ndarray
            low-pass filter for reconstruction
        hr : np.ndarray
            high-pass filter for reconstruction
    """

    w = pywt.Wavelet(kind)
    ld = np.array(w.dec_lo)
    hd = np.array(w.dec_hi)
    lr = np.array(w.rec_lo)
    hr = np.array(w.rec_hi)

    return ld, hd, lr, hr


def dwt_max_level(data_length, filter_length):
    """Return the highest achievable DWT level, given the provided data- and filter lengths
    
    Parameters
    ----------
        data_length: int
            length of the data `x`, on which the DWT will be performed
        filter_length: int
            length of the lowpass which will be used to perform the DWT
    
    Returns
    -------
        n_max: int
    """
    n_max = int(np.floor(np.log2(data_length / (filter_length - 1))))
    return n_max


def wave_dec(x, Ux, lowpass, highpass, n=-1):
    """
    Multilevel discrete wavelet transformation of time-series x with uncertainty Ux.

    Parameters:
    -----------
        x : np.ndarray
            input signal
        Ux : float or np.ndarray
            float: standard deviation of white noise in x
            1D-array: point-wise standard uncertainties of non-stationary white noise
        lowpass : np.ndarray
            decomposition low-pass for wavelet_block
        highpass : np.ndarray
            decomposition high-pass for wavelet_block
        n : int, optional (default: -1)
            consecutive repetitions of wavelet_block
            user is warned, if it is not possible to reach the specified depth
            use highest possible level if set to -1 (default)

    Returns:
    --------
        coeffs : list of arrays
            order of arrays within list is:
            [cAn, cDn, cDn-1, ..., cD2, cD1]
        Ucoeffs : list of arrays
            uncertainty of coeffs, same order as coeffs
        original_length : int
            equals to len(x)
            necessary to restore correct length 
    """

    # check if depth is reachable
    max_depth = dwt_max_level(x.size, lowpass.size)
    if n > max_depth:
        raise UserWarning(
            "Will run into trouble, max_depth = {MAX_DEPTH}, but you specified {DEPTH}. Consider reducing the depth-to-be-achieved or prolong the input signal.".format(
                DEPTH=n, MAX_DEPTH=max_depth
            )
        )
    elif n == -1:
        n = max_depth

    c_approx = x
    Uc_approx = Ux

    original_length = len(x)
    coeffs = []
    Ucoeffs = []

    for level in range(n):

        # execute wavelet block
        c_approx, Uc_approx, c_detail, Uc_detail, _ = dwt(
            c_approx, Uc_approx, lowpass, highpass
        )

        # save result
        coeffs.insert(0, c_detail)
        Ucoeffs.insert(0, Uc_detail)
        if level + 1 == n:  # save the details when in last level
            coeffs.insert(0, c_approx)
            Ucoeffs.insert(0, Uc_approx)

    return coeffs, Ucoeffs, original_length


def wave_dec_realtime(x, Ux, lowpass, highpass, n=1, level_states=None):
    """
    Multilevel discrete wavelet transformation of time-series x with uncertainty Ux.
    Similar to :func:`wave_dec`, but allows to start from the internal_state of a previous call.

    Parameters:
    -----------
        x : np.ndarray
            input signal
        Ux : float or np.ndarray
            float: standard deviation of white noise in x
            1D-array: point-wise standard uncertainties of non-stationary white noise
        lowpass : np.ndarray
            decomposition low-pass for wavelet_block
        highpass : np.ndarray
            decomposition high-pass for wavelet_block
        n : int, optional (default: 1)
            consecutive repetitions of wavelet_block
            There is no maximum level in continuos wavelet transform, so the default is n=1. 
        level_states : dict, optional (default: None)
            internal state from previous call

    Returns:
    --------
        coeffs : list of arrays
            order of arrays within list is:
            [cAn, cDn, cDn-1, ..., cD2, cD1]
        Ucoeffs : list of arrays
            uncertainty of coeffs, same order as coeffs
        original_length : int
            equals to len(x)
            necessary to restore correct length 
        level_states : dict
            last internal state
    """
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
        i_n = i0 // 2 ** level
        subsample_start = (i_n + 1) % 2

        # execute wavelet block
        if len(c_approx) > 0:
            c_approx, Uc_approx, c_detail, Uc_detail, level_states[level] = dwt(
                c_approx,
                Uc_approx,
                lowpass,
                highpass,
                realtime=True,
                states=level_states[level],
                subsample_start=subsample_start,
            )
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
    level_states["counter"] = (level_states["counter"] + len(x)) % 2 ** n

    return coeffs, Ucoeffs, original_length, level_states


def wave_rec(coeffs, Ucoeffs, lowpass, highpass, original_length=None):
    """
    Multilevel discrete wavelet reconstruction of coefficients from levels back into time-series.

    Parameters:
    -----------
        coeffs : list of arrays
            order of arrays within list is:
            [cAn, cDn, cDn-1, ..., cD2, cD1]
            where:

            * cAi: approximation coefficients array from i-th level
            * cDi: detail coefficients array from i-th level
        Ucoeffs : list of arrays
            uncertainty of coeffs, same order as coeffs
        lowpass : np.ndarray
            reconstruction low-pass for wavelet_block
        highpass : np.ndarray
            reconstruction high-pass for wavelet_block
        original_length : int, optional (default: None)
            necessary to restore correct length of original time-series
    
    Returns
    -------
        x : np.ndarray
            reconstructed signal
        Ux : np.ndarray
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

        # execute inv_dwt
        c_approx, U_approx, _ = inv_dwt(
            c_approx, U_approx, c_detail, U_detail, lowpass, highpass
        )

    # bring to original length (does nothing if original_length == None)
    x = c_approx[:original_length]
    Ux = U_approx[:original_length]

    return x, Ux
