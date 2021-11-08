"""A collection of test signals which can be used to simulate dynamic measurements

This module contains the following functions:

* :func:`GaussianPulse`: Generate a Gaussian pulse at t0 with height m0 and std sigma
* :func:`multi_sine`: Generate a multi-sine signal as summation of single sine signals
* :func:`rect`: Rectangular signal of given height and width :math:`t_1 - t_0`
* :func:`shocklikeGaussian`: Generate a signal that resembles a shock excitation as a
  Gaussian
* :func:`sine`: Generate a sine signal
* :func:`squarepulse`: Generates a series of rect functions to represent a square
  pulse signal
"""

__all__ = [
    "shocklikeGaussian",
    "GaussianPulse",
    "rect",
    "squarepulse",
    "corr_noise",
    "sine",
    "multi_sine",
]

import itertools

import numpy as np
from numpy import array, corrcoef, diff, sqrt, sum
from scipy.signal import periodogram
from scipy.special import comb

from .noise import white_gaussian


def shocklikeGaussian(time, t0, m0, sigma, noise=0.0):
    """Generate a signal that resembles a shock excitation as a Gaussian

    The main shock is followed by a smaller Gaussian of opposite sign.

    Parameters
    ----------
    time : np.ndarray of shape (N,)
        time instants (equidistant)
    t0 : float
        time instant of signal maximum
    m0 : float
         signal maximum
    sigma : float
         std of main pulse
    noise : float, optional
        std of simulated signal noise

    Returns
    -------
    x : np.ndarray of shape (N,)
        signal amplitudes at time instants
    """

    x = (
        -m0
        * (time - t0)
        / sigma
        * np.exp(0.5)
        * np.exp(-((time - t0) ** 2) / (2 * sigma ** 2))
    )
    if noise > 0:
        x = white_gaussian(len(x), x, noise)
    return x


def GaussianPulse(time, t0, m0, sigma, noise=0.0):
    """Generate a Gaussian pulse at t0 with height m0 and std sigma

    Parameters
    ----------
    time : np.ndarray of shape (N,)
        time instants (equidistant)
    t0 : float
       time instant of signal maximum
    m0 : float
        signal maximum
    sigma : float
        std of pulse
    noise : float, optional
        std of simulated signal noise

    Returns
    -------
    x : np.ndarray of shape (N,)
        signal amplitudes at time instants
    """

    x = m0 * np.exp(-((time - t0) ** 2) / (2 * sigma ** 2))
    if noise > 0:
        x = white_gaussian(len(x), x, noise)
    return x


def rect(time, t0, t1, height=1, noise=0.0):
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
    noise :float or numpy.ndarray of shape (N,), optional
        float: standard deviation of additive white gaussian noise
        ndarray: user-defined additive noise

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
            x = white_gaussian(len(x), x, noise)
    elif isinstance(noise, np.ndarray):
        if x.size == noise.size:
            x += noise
        else:
            raise ValueError("Mismatching sizes of x and noise.")
    else:
        raise NotImplementedError(
            "The given noise is neither of type float nor numpy.ndarray. "
        )

    return x


def squarepulse(time, height, numpulse=4, noise=0.0):
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
        x = white_gaussian(len(x), x, noise)
    return x


def sine(time, amp=1.0, freq=1.0, noise=0.0):
    r"""Generate a batch of a sine signal with normally distributed noise

    Parameters
    ----------
    time : np.ndarray of shape (N,)
        time instants
    amp : float, optional
         amplitude of the sine (default = 1.0)
    freq : float, optional
         frequency of the sine in Hz (default = 1.0)
    noise : float, optional
        std of simulated signal noise (default = 0.0)

    Returns
    -------
    x : np.ndarray of shape (N,)
        signal amplitude at time instants
    """
    # Design the sine signal according to e.g.
    #   https://de.wikipedia.org/wiki/Sinuston#Mathematischer_Hintergrund
    x = amp * np.sin(2 * np.pi * freq * time)
    if noise:  # noise = 0.0 (default)  is equivalent to noise = False here
        x = white_gaussian(len(x), x, noise)
    return x


def multi_sine(time, amps, freqs, noise=0.0):
    r"""Generate a batch of a summation of sine signals with normally distributed noise

    Parameters
    ----------
    time : np.ndarray of shape (N,)
        time instants
    amps : list or np.ndarray of shape (M,) of floating point values
        amplitudes of the sine signals
    freqs : list or np.ndarray of shape (M,) of floating point values
        frequencies of the sine signals in Hz
    noise : float, optional
        std of simulated signal noise (default = 0.0)

    Returns
    -------
    x : np.ndarray of shape (N,)
        signal amplitude at time instants
    """

    x = np.zeros_like(time)
    for amp, freq in zip(amps, freqs):
        x += sine(time=time, amp=amp, freq=freq, noise=0.0)
    if noise:
        x = white_gaussian(len(x), x, noise)
    return x


class corr_noise:
    """Base class for generation of a correlated noise process"""

    def __init__(self, w, sigma, seed=None):
        self.w = w
        self.sigma = sigma
        self.rst = np.random.RandomState(seed)
        self.Cw = None
        self.noise = None
        self.Sigma = None

    def calc_noise(self, N=100):
        z = self.rst.standard_normal(N + 4)
        noise = (
            diff(
                diff(
                    diff(diff(z * self.w ** 4) - 4 * z[1:] * self.w ** 3)
                    + 6 * z[2:] * self.w ** 2
                )
                - 4 * z[3:] * self.w
            )
            + z[4:]
        )
        self.Cw = sqrt(sum([comb(4, el) ** 2 * self.w ** (2 * el) for el in range(5)]))
        self.noise = noise * self.sigma / self.Cw
        return self.noise

    def calc_noise2(self, N=100):
        P = np.ceil(1.5 * N)
        NT = self.rst.standard_normal(P) * self.sigma
        STD = np.zeros(21)
        STD[10] = 1.0
        for _ in itertools.repeat(None, 5):
            NTtmp = NT.copy()
            NT[:-1] = NT[:-1] + self.w * NTtmp[1:]
            NT[-1] = NT[-1] + self.w * NTtmp[-1]
            NT[1:] = NT[1:] + self.w * NTtmp[:-1]
            NT[0] = NT[0] + self.w * NTtmp[-1]
            STDtmp = STD.copy()
            STD[1:] = STD[1:] + self.w * STDtmp[:-1]
            STD[:-1] = STD[:-1] + self.w * STDtmp[1:]
        NT /= np.linalg.norm(STD)
        self.noise = NT[:N]
        self.Cw = sqrt(sum([comb(4, el) ** 2 * self.w ** (2 * el) for el in range(5)]))
        return self.noise

    def calc_autocorr(self, lag=10):
        return array(
            [1]
            + [corrcoef(self.noise[:-i], self.noise[i:])[0, 1] for i in range(1, lag)]
        )

    def calc_cov(self):
        def cw(k):
            if np.abs(k) > 4:
                return 0
            c = sum(
                [
                    comb(4, el)
                    * comb(4, np.abs(k) + el)
                    * self.w ** (np.abs(k) + 2 * el)
                    for el in range(5 - np.abs(k))
                ]
            )
            return c / self.Cw ** 2

        N = len(self.noise)
        Sigma = np.zeros((N, N))
        for m in range(N):
            for n in range(m, N):
                Sigma[m, n] = self.sigma ** 2 * cw(n - m)
        self.Sigma = Sigma + Sigma.T - np.diag(np.diag(Sigma))
        return self.Sigma

    def calc_psd(self, noise=None, Fs=1.0, **kwargs):
        if isinstance(noise, np.ndarray):
            return periodogram(noise, fs=Fs, **kwargs)
        return periodogram(self.noise, fs=Fs, **kwargs)
