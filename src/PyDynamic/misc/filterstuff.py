# -*- coding: utf-8 -*-
r"""
The :mod:`PyDynamic.misc.filterstuff` module is a collection of methods which are
related to filter design.

This module contains the following functions:

* :func:`db`: Calculation of decibel values :math:`20\log_{10}(x)` for a vector of
  values
* :func:`ua`: Shortcut for calculation of unwrapped angle of complex values
* :func:`grpdelay`: Calculation of the group delay of a digital filter
* :func:`mapinside`: Maps the roots of polynomial with coefficients :math:`a`
  to the unit circle
* :func:`kaiser_lowpass`: Design of a FIR lowpass filter using the window technique
  with a Kaiser window.
* :func:`isstable`: Determine whether a given IIR filter is stable
* :func:`savitzky_golay`: Smooth (and optionally differentiate) data with a
  Savitzky-Golay filter

"""

import numpy as np

__all__ = [
    "db",
    "ua",
    "grpdelay",
    "mapinside",
    "kaiser_lowpass",
    "isstable",
    "savitzky_golay",
]


def db(vals):
    """Calculation of decibel values :math:`20\log_{10}(x)` for a vector of values"""
    return 20 * np.log10(np.abs(vals))


def ua(vals):
    """Shortcut for calculation of unwrapped angle of complex values"""
    return np.unwrap(np.angle(vals))


def grpdelay(b, a, Fs, nfft=512):
    """Calculation of the group delay of a digital filter
   
    Parameters
    ----------
        b: ndarray
            IIR filter numerator coefficients
        a: ndarray
            IIR filter denominator coefficients
        Fs: float
            sampling frequency of the filter
        nfft: int
            number of FFT bins

    Returns
    -------
        group_delay: np.ndarray
            group delay values
        frequencies: ndarray
            frequencies at which the group delay is calculated

    References
    ----------
    * Smith, online book [Smith]_

    """

    Na = len(a) - 1
    Nb = len(b) - 1

    c = np.convolve(b, a[::-1])  # c(z) = b(z)*a(1/z)*z^(-oa)
    cr = c * np.arange(Na + Nb + 1)  # derivative of c wrt 1/z
    num = np.fft.fft(cr, 2 * nfft)
    den = np.fft.fft(c, 2 * nfft)
    tol = 1e-12

    polebins = np.nonzero(abs(den) < tol)
    for p in polebins:
        num[p] = 0.0
        den[p] = 1.0

    gd = np.real(num / den) - Na

    f = np.arange(0.0, 2 * nfft - 1) / (2 * nfft) * Fs
    f = f[: nfft + 1]
    gd = gd[: len(f)]
    return gd, f


def mapinside(a):
    """ Maps the roots of polynomial to the unit circle.

    Maps the roots of polynomial with coefficients :math:`a` to the unit circle.
    
    Parameters
    ----------
        a: ndarray
           polynomial coefficients
    Returns
    -------
        a: ndarray
           polynomial coefficients with all roots inside or on the unit circle
    """
    v = np.roots(a)
    inds = np.nonzero(abs(v) > 1)
    v[inds] = 1 / np.conj(v[inds])
    return np.poly(v)


def kaiser_lowpass(L, fcut, Fs, beta=8.0):
    """ Design of a FIR lowpass filter using the window technique with a
    Kaiser window.

    This method uses a Kaiser window. Filters of that type are often used as
    FIR low-pass filters due to their linear phase.
    
    Parameters
    ----------
        L: int
           filter order (window length)
        fcut: float
              desired cut-off frequency
        Fs: float
            sampling frequency
        beta: float
              scaling parameter for the Kaiser window
    Returns
    -------
        blow: ndarray
              FIR filter coefficients
        shift: int
               delay of the filter (in samples)
    
    """
    from scipy.signal import firwin

    if np.mod(L, 2) == 0:
        L = L + 1
    blow = firwin(L, 2 * fcut / Fs, window=("kaiser", beta))
    shift = L / 2
    return blow, shift


def isstable(b, a, ftype="digital"):
    """Determine whether `IIR filter (b,a)` is stable

    Determine whether `IIR filter (b,a)` is stable by checking roots of the
    polynomial ´a´.
    
    Parameters
    ----------
        b: ndarray
            filter numerator coefficients
        a: ndarray
            filter denominator coefficients
        ftype: string
            type of filter (`digital` or `analog`)
    Returns
    -------
        stable: bool
            whether filter is stable or not
                
    """
    v = np.roots(a)
    if ftype == "digital":
        return not np.any(np.abs(v) > 1.0)
    elif ftype == "analog":
        return not np.any(np.real(v) < 0)
    else:
        raise ValueError(f"isstable: filter type 'ftype={ftype}'unknown. Please "
                         f"check documentation and choose a valid assignment for "
                         f"'ftype' or leave out to fallback to default.")


def savitzky_golay(y, window_size, order, deriv=0, delta=1.0):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter

    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Source obtained from scipy cookbook (online), downloaded 2013-09-13

    Parameters
    ----------
        y: ndarray, shape (N,)
           the values of the time history of the signal
        window_size: int
           the length of the window. Must be an odd integer number
        order: int
           the order of the polynomial used in the filtering. Must be less
           then `window_size` - 1.
        deriv: int, optional
            The order of the derivative to compute.  This must be a
            nonnegative integer.  The default is 0, which means to filter
            the data without differentiating.
        delta: float, optional
            The spacing of the samples to which the filter will be applied.
            This is only used if deriv > 0. This includes a factor
            :math:`n! / h^n`, where :math:`n` is represented by `deriv` and
            :math:`1/h` by `delta`.

    Returns
    -------
         ys: ndarray, shape (N,)
            the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.


    References
    ----------
    * Savitzky et al. [Savitzky]_
    * Numerical Recipes [NumRec]_

    """
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat(
        [[k ** i for i in order_range] for k in range(-half_window, half_window + 1)]
    )
    m = np.linalg.pinv(b).A[deriv] * factorial(deriv) / delta ** deriv
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")
