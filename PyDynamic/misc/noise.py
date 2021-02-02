# -*- coding: utf-8 -*-
"""Collection of noise-signals

This module contains the following functions:

* :func:`get_alpha`: normal distributed signal amplitudes with equal power
  spectral density
* :func:`power_law_noise`: normal distributed signal amplitudes with power spectrum
  `:math:f^\alpha`
* :func:`power_law_acf`: (theoretical) autocorrelation function of power law noise
* :func:`ARMA`: autoregressive moving average noise process
"""

import numpy as np
from scipy.linalg import toeplitz

__all__ = ["get_alpha", "white_gaussian", "power_law_noise", "power_law_acf", "ARMA"]


# define an alpha for every color
colors = {"violet": 2, "blue": 1, "white": 0, "pink": -1, "red": -2, "brown": -2}


def get_alpha(color_value=0):
    """
    Translate a color (given as string) into an exponent alpha or directly
    hand through a given numeric value of alpha.

    Parameters
    ----------
        color_value: str, int or float
            if string -> check against known colornames -> return alpha
            if numeric -> directly return value

    Returns
    -------
        alpha: float
    """

    if isinstance(color_value, str):
        if color_value in colors.keys():
            alpha = colors[color_value]
        else:
            raise NotImplementedError(
            "Specified color ({COLOR}) of noise is not available. " \
            "Please choose from {COLORS} or define alpha directly." \
            .format(COLOR=color_value, COLORS='/'.join(colors.keys())))

    elif isinstance(color_value, (float, int)):
        alpha = color_value

    else:
        raise IOError("No valid string or numeric value for alpha given")

    return float(alpha)


def white_gaussian(N, mean=0, std=1):
    return np.random.normal(loc=mean, scale=std, size=N)


def power_law_noise(N=None, w=None, color_value="white", mean=0.0, std=1.0):
    """
    Generate colored noise by
    * generate white gaussian noise
    * multiplying its Fourier-transform with f^(alpha/2)
    * inverse Fourier-transform to yield the colored/correlated noise
    * further adjustments to fit to specified mean/std

    based on [Zhivomirov2018](A Method for Colored Noise Generation)

    Parameters
    ----------
        N: int
            length of noise to be generated
        w: numpy.ndarray
            user-defined white noise
            if provided, `N` is ignored!
        color_value: str, int or float
            if string -> check against known colornames
            if numeric -> used as alpha to shape PSD
        mean: float
            mean of the output signal
        std: float
            standard deviation of the output signal

    Returns
    -------
        w_filt: filtered noise signal
    """

    if (N is not None) and (w is not None):
        raise UserWarning("You specified N and w. Ignoring N.")

    # draw white gaussian noise, or take the provided w
    if isinstance(w, np.ndarray):
        N = len(w)
    else:
        w = white_gaussian(N)

    # get alpha either directly or from color-string
    alpha = get_alpha(color_value)

    # (real) fourier transform to get spectrum
    W = np.fft.rfft(w)

    # get index of frequencies
    # note:
    # * this gives [1., 2., 3., ..., N+1] (in accordance with [Zhivomirov2018])
    # * ==> not W_filt ~ f^alpha, but rather W_filt ~ k^alpha
    steps = N // 2 + 1
    k = np.linspace(0, steps, steps) + 1

    # generate the filtered spectrum by multiplication with f^(alpha/2)
    W_filt = W * np.power(k, alpha / 2)

    # calculate the filtered time-series (inverse fourier of modified spectrum)
    w_filt = np.fft.irfft(W_filt, N)

    # adjust to given mean + std
    w_filt = (w_filt - np.mean(w_filt)) / np.std(w_filt)
    w_filt = mean + std * w_filt

    return w_filt


def power_law_acf(N, color_value="white", std=1.0):
    """
    Return the theoretic right-sided autocorrelation (Rww) of different colors of noise. 

    Colors of noise are defined to have a power spectral density (Sww) proportional to `f^\alpha`.
    Sww and Rww form a Fourier-pair. Therefore Rww = ifft(Sww).
    """
    # process the arguments
    alpha = get_alpha(color_value)

    # get index of frequencies (see notes of same line at power_law_noise() )
    k = np.linspace(0, N, N) + 1

    # generate and transform the power spectral density Sww
    Sww = np.power(k, alpha)

    # normalize Sww to have the same overall power as the white-noise-PSD it is transformed from
    #Sww = Sww / np.sum(Sww) * len(k)   # probably unnecessary because of later normalization of Rww

    # inverse Fourier-transform to get Autocorrelation from PSD/Sww
    Rww = np.fft.irfft(Sww)
    Rww = std**2 * Rww / Rww[0]           # This normalization ensures the given standard-deviation

    return Rww[:N]


def ARMA(length, phi=0.0, theta=0.0, std=1.0):
    r"""
    Generate time-series of a predefined ARMA-process based on this equation:
    :math:`\sum_{j=1}^{\min(p,n-1)} \phi_j \epsilon[n-j] + \sum_{j=1}^{\min(q,n-1)} \theta_j w[n-j]`
    where w is white gaussian noise. Equation and algorithm taken from [Eichst2012]_ .

    Parameters
    ----------
    length: int
        how long the drawn sample will be
    phi: float, list or numpy.ndarray, shape (p, )
        AR-coefficients
    theta: float, list or numpy.ndarray
        MA-coefficients
    std: float
        std of the gaussian white noise that is feeded into the ARMA-model

    Returns
    -------
    e: numpy.ndarray, shape (length, )
       time-series of the predefined ARMA-process

    References
    ----------
        * Eichstädt, Link, Harris and Elster [Eichst2012]_
    """

    # convert to numpy.ndarray
    if isinstance(phi, (float, int)):
        phi = np.array([phi])
    elif isinstance(phi, list):
        phi = np.array(phi)

    if isinstance(theta, (float, int)):
        theta = np.array([theta])
    elif isinstance(theta, list):
        theta = np.array(theta)

    # initialize e, w
    w = np.random.normal(loc=0, scale=std, size=length)
    e = np.zeros_like(w)

    # define shortcuts
    p = len(phi)
    q = len(theta)

    # iterate series over time
    for n, wn in enumerate(w):
        min_pn = min(p, n)
        min_qn = min(q, n)
        e[n] = np.sum(phi[:min_pn].dot(e[n-min_pn:n])) + np.sum(theta[:min_qn].dot(w[n-min_qn:n])) + wn

    return e
