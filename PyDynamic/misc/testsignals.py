# -*- coding: utf-8 -*-
"""Collection of test signals which can be used to simulate dynamic measurements
and test methods.

This module contains the following functions:
* shocklikeGaussian: signal that resembles a shock excitation as a Gaussian
followed by a smaller Gaussian of opposite sign
* GaussianPulse: Generates a Gaussian pulse at t0 with height m0 and std sigma
* rect: Rectangular signal of given height and width t1-t0
* squarepulse: Generates a series of rect functions to represent a square
pulse signal
"""

import numpy as np
from numpy import diff, sqrt, sum, array, corrcoef
from scipy.signal import periodogram
from scipy.special import comb

__all__ = ['shocklikeGaussian', 'GaussianPulse', 'rect', 'squarepulse']


def shocklikeGaussian(time, t0, m0, sigma, noise = 0.0):
    """Generates a signal that resembles a shock excitation as a Gaussian followed by a smaller Gaussian of opposite sign.

    Parameters
    ----------
        time : np.ndarray of shape (N,)
            time instants (equidistant)
        t0: float
            time instant of signal maximum
        m0: float
            signal maximum
        sigma: float
            std of main pulse
        noise: float, optional
            std of simulated signal noise

    Returns
    -------
        x: np.ndarray of shape (N,)
            signal amplitudes at time instants

    """

    x = -m0*(time-t0)/sigma * np.exp(0.5) * np.exp(-(time-t0)**2/(2*sigma**2))
    if noise > 0:
        x += np.random.randn(len(time)) * noise
    return x


def GaussianPulse(time, t0, m0, sigma, noise = 0.0):
    """Generates a Gaussian pulse at t0 with height m0 and std sigma

    Parameters
    ----------
        time: np.ndarray of shape (N,)
            time instants (equidistant)
        t0 : float
            time instant of signal maximum
        m0 : float
            signal maximum
        sigma : float
            std of pulse
        noise: float, optional
            std of simulated signal noise

    Returns
    -------
        x : np.ndarray of shape (N,)
            signal amplitudes at time instants
    """

    x = m0 * np.exp(-(time - t0) ** 2 / (2 * sigma ** 2))
    if noise > 0:
        x = x + np.random.randn(len(time)) * noise
    return x


def rect(time, t0, t1, height = 1, noise = 0.0):
    """Rectangular signal of given height and width t1-t0

    Parameters
    ----------
        time : np.ndarray of shape (N,)
            time instants (equidistant)
        t0 : float
            time instant of rect lhs
        t1 : float
            time instant of rect rhs
        height : float
            signal maximum
        noise :float, optional
            std of simulated signal noise

    Returns
    -------
        x : np.ndarray of shape (N,)
            signal amplitudes at time instants
    """

    x = np.zeros((len(time),))
    x[np.nonzero(time > t0)] = height
    x[np.nonzero(time > t1)] = 0.0

    # add the noise
    if isinstance(noise, float):
        if noise > 0:
            x = x + np.random.randn(len(time)) * noise
    elif isinstance(noise, np.ndarray):
        if x.size == noise.size:
            x = x + noise
        else:
            raise ValueError("Mismatching sizes of x and noise.")
    else:
        raise NotImplementedError("The given noise is neither of type float nor numpy.ndarray. ")
    return x


def squarepulse(time, height, numpulse = 4, noise = 0.0):
    """Generates a series of rect functions to represent a square pulse signal

    Parameters
    ----------
        time : np.ndarray of shape (N,)
            time instants
        height : float
            height of the rectangular pulses
        numpulse : int
            number of pulses
        noise : float, optional
            std of simulated signal noise

    Returns
    -------
        x : np.ndarray of shape (N,)
            signal amplitude at time instants
    """
    width = (time[-1] - time[0]) / (2 * numpulse + 1)  # width of each individual rect
    x = np.zeros_like(time)
    for k in range(numpulse):
        x += rect(time, (2 * k + 1) * width, (2 * k + 2) * width, height)
    if noise > 0:
        x += np.random.randn(len(time)) * noise
    return x


class corr_noise(object):
    """Base class for generation of a correlated noise process."""

    def __init__(self, w, sigma, seed=None):
        self.w = w
        self.sigma = sigma
        self.rst = np.random.RandomState(seed)

        # define a beta for every color
        self.colors = {"violet": -2,
                       "blue":   -1,
                       "white":   0,
                       "pink":    1,
                       "red":     2,
                       "brown":   2 }

    def calc_noise(self, N = 100):
        z = self.rst.randn(N + 4)
        noise = diff(diff(diff(diff(z * self.w ** 4) - 4 * z[1:] * self.w ** 3) + 6 * z[2:] * self.w ** 2) - 4 * z[3:] * self.w) + z[4:]
        self.Cw = sqrt(sum([comb(4, l) ** 2 * self.w ** (2 * l) for l in range(5)]))
        self.noise = noise * self.sigma / self.Cw
        return self.noise

    def calc_noise2(self, N = 100):
        P = np.ceil(1.5 * N)
        NT = self.rst.randn(P) * self.sigma
        STD = np.zeros(21)
        STD[10] = 1.0
        for counter in range(5):
            NTtmp = NT.copy()
            NT[:-1] = NT[:-1] + self.w * NTtmp[1:]
            NT[-1] = NT[-1] + self.w * NTtmp[-1]
            NT[1:] = NT[1:] + self.w * NTtmp[:-1]
            NT[0] = NT[0] + self.w * NTtmp[-1]
            STDtmp = STD.copy()
            STD[1:] = STD[1:] + self.w * STDtmp[:-1]
            STD[:-1] = STD[:-1] + self.w * STDtmp[1:]
        NT = NT / np.linalg.norm(STD)
        self.noise = NT[:N]
        self.Cw = sqrt(sum([comb(4, l) ** 2 * self.w ** (2 * l) for l in range(5)]))
        return self.noise

    def calc_autocorr(self, lag = 10):
        return array([1] + [corrcoef(self.noise[:-i], self.noise[i:])[0, 1] for i in range(1, lag)])

    def calc_cov(self):
        def cw(k):
            if np.abs(k) > 4: return 0
            c = sum([comb(4, l) * comb(4, np.abs(k) + l) * self.w ** (np.abs(k) + 2 * l) for l in range(5 - np.abs(k))])
            return c / self.Cw ** 2

        N = len(self.noise)
        Sigma = np.zeros((N, N))
        for m in range(N):
            for n in range(m, N):
                Sigma[m, n] = self.sigma ** 2 * cw(n - m)
        self.Sigma = Sigma + Sigma.T - np.diag(np.diag(Sigma))
        return self.Sigma

    def calc_psd(self, noise = None, Fs = 1.0, **kwargs):
        if isinstance(noise, np.ndarray):
            return periodogram(noise, fs = Fs, **kwargs)
        else:
            return periodogram(self.noise, fs = Fs, **kwargs)

    def getBeta(self, beta, colorString):
        # raise warning, if beta and color are non-None
        if (beta is not None) and (colorString not in [None, "white"]):
            raise UserWarning("You have specified a colorString and beta. Only beta will be considered, the colorString is ignored!")

        # define beta from color-string, if no beta-argument was handed over
        if beta is None:
            if colorString in self.colors.keys():
                beta = self.colors[colorString]
            else: raise NotImplementedError(
                "Specified color ({COLOR}) of noise is not available. Please choose from {COLORS} or define beta directly.".format(COLOR=colorString, COLORS='/'.join(self.colors.keys())))
        return beta

    def theoretic_covariance_colored_noise(self, N = None, color = "white", beta = None):
        """
        Return the theoretic autocovariance-matrix (Rww) of different colors of noise. If "beta" is provided, "color"-argument is ignored.

        Colors of noise are defined to have a power spectral density (Sww) proportional to `1/f^beta`.
        Sww and Rww form a Fourier-pair. Therefore Rww = ifft(Sww).
        """
        # process the arguments
        if N == None: N = len(self.w)
        beta = self.getBeta(beta, color)

        # generate frequencies
        freq = np.fft.fftfreq(2*N)

        # generate and transform the power spectral density Sww
        Sww = 1.0 / np.power(np.abs(freq), beta)
        Sww[0] = 1                             #  Sww[0] is NaN for positive betas, FIXME: Setting it to 1 is suitable because ... ?

        Rww = np.real(np.fft.ifft(Sww))
        Rww = self.sigma**2 * Rww / Rww[0]     # This normalization ensures the given standard-deviation

        return Rww                             # attention, Rww has length 2*N, this allows for cyclic repetition --> Rxx[2N] == Rxx[-1]

        # build matrix from this result
        #Rww_matrix = np.vstack([np.roll(Rww,shift)[0:N] for shift in range(N)])
        #return Rww_matrix

    def colored_noise(self, beta = None, color = "white"):
        """
        Generate colored noise by
        * taking the (assumingly) white noise `self.w`
        * dividing its Fourier-transform with f^(beta/2)
        * inverse Fourier-transform to yield the colored/correlated noise
        """

        beta = self.getBeta(beta, color)

        sp   = np.fft.fft(self.w)
        freq = np.fft.fftfreq(len(self.w))

        # generate the filtered spectrum by multiplication by f^(k/2) (should do the same as applying f^beta to the PSD)
        sp_filt = sp / np.power(np.abs(freq), beta/2)

        ## compensate division by zero errors for beta<0, also sets signal to zero-mean
        sp_filt[0] = 0

        # calculate the filtered time-series
        w_filt = np.real(np.fft.ifft(sp_filt))

        # back to sigma variance
        w_filt = self.sigma * w_filt / np.std(w_filt)

        return w_filt
