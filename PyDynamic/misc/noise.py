# -*- coding: utf-8 -*-
"""Collection of noise-signals

This module contains the following functions:
* white_gaussian: normal distributed signal amplitudes with equal power spectral density
* power_law_noise: normal distributed signal amplitudes with power spectrum `~ f^\alpha`
* power_law_acf: (theoretical) autocorrelation function of power law noise
"""

import numpy as np

__all__ = ['white_gaussian', 'power_law_noise', 'power_law_acf']

import numpy as np


# define an alpha for every color
colors = {"violet": 2,
          "blue":  1,
          "white": 0,
          "pink": -1,
          "red":  -2,
          "brown": -2}


def getAlpha(alpha, colorString):
    # raise warning, if alpha and color are non-None
    if (alpha is not None) and (colorString not in [None, "white"]):
        raise UserWarning("You have specified a colorString and alpha. Only alpha will be considered, the colorString is ignored!")

    # define alpha from color-string, if no alpha-argument was handed over
    if alpha is None:
        if colorString in colors.keys():
            alpha = colors[colorString]
        else: raise NotImplementedError(
            "Specified color ({COLOR}) of noise is not available. Please choose from {COLORS} or define alpha directly.".format(COLOR=colorString, COLORS='/'.join(colors.keys())))
    return alpha

def white_gaussian(N, mean = 0, std = 1):
    return np.random.normal(loc=mean, scale = std, size = N)

def power_law_noise(N = None, w = None, alpha = None, color = "white", mean = 0, std = 1):
    """
    Generate colored noise by
    * generate white gaussian noise
    * multiplying its Fourier-transform with f^(alpha/2)
    * inverse Fourier-transform to yield the colored/correlated noise
    * further adjustments to fit to specified mean/std

    based on [Zhivomirov2018](A Method for Colored Noise Generation)
    """

    # draw white gaussian noise, or take the provided w
    if isinstance(w, np.ndarray):
        N = len(w)
        mean = np.mean(w)
        std  = np.std(w)
    else:
        w = white_gaussian(N)

    # get alpha either directly or from color-string
    alpha = getAlpha(alpha, color)

    # (real) fourier transform to get spectrum
    sp   = np.fft.rfft(w)

    # get index of frequencies
    # note: 
    # * this gives [1., 2., 3., ..., N+1] (in accordance with Zhivomirov2018)
    # * ==> not ~ f^alpha, but rather ~ (k+1)^alpha (with a zero-based index k)
    # * this could be a point for further discussion
    freq = np.fft.rfftfreq(N, d=1/N) + 1

    # generate the filtered spectrum by multiplication with k^(alpha/2)
    sp_filt = sp * np.power(freq, alpha/2)

    # calculate the filtered time-series (inverse fourier of modified spectrum)
    w_filt = np.fft.irfft(sp_filt, N)

    # adjust to given mean + std
    w_filt = mean + std * (w_filt - np.mean(w_filt)) / np.std(w_filt)

    return w_filt

def power_law_acf(N, color = "white", alpha = None, mean = 0, std = 1):
    """
    Return the theoretic autocovariance-matrix (Rww) of different colors of noise. If "alpha" is provided, "color"-argument is ignored.

    Colors of noise are defined to have a power spectral density (Sww) proportional to `f^\alpha`.
    Sww and Rww form a Fourier-pair. Therefore Rww = ifft(Sww).
    """
    # process the arguments
    alpha = getAlpha(alpha, color)

    # get index of frequencies (see notes of same line at power_law_noise() )
    freq = np.fft.rfftfreq(2*N, d=1/(2*N)) + 1

    # generate and transform the power spectral density Sww
    Sww = np.power(freq, alpha)

    # normalize Sww to have the same overall power as the white-noise-PSD it is transformed from
    Sww = Sww / np.sum(Sww) * len(freq)   # probably unnecessary because of later normalization of Rww

    # inverse Fourier-transform to get Autocorrelation from PSD/Sww
    Rww = np.fft.irfft(Sww, 2*N)
    Rww = std**2 * Rww / Rww[0]           # This normalization ensures the given standard-deviation

    return Rww[:N]
