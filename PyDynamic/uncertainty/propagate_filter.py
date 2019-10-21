# -*- coding: utf-8 -*-
"""
This module contains functions for the propagation of uncertainties through
the application of a digital filter using the GUM approach.

This modules contains the following functions:

* *FIRuncFilter*: Uncertainty propagation for signal y and uncertain FIR
  filter theta
* *IIRuncFilter*: Uncertainty propagation for the signal x and the uncertain
  IIR filter (b,a)

# Note: The Elster-Link paper for FIR filters assumes that the autocovariance
is known and that noise is stationary!

"""
import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import lfilter, tf2ss

__all__ = ['FIRuncFilter', 'IIRuncFilter']


def FIRuncFilter(y, sigma_noise, theta, Utheta=None, shift=0, blow=None):
    """Uncertainty propagation for signal y and uncertain FIR filter theta

    Parameters
    ----------
        y: np.ndarray
            filter input signal
        sigma_noise: float or np.ndarray
            when float then standard deviation of white noise in y; when
            ndarray then point-wise standard uncertainties
        theta: np.ndarray
            FIR filter coefficients
        Utheta: np.ndarray
            covariance matrix associated with theta
        shift: int
            time delay of filter output signal (in samples)
        blow: np.ndarray
            optional FIR low-pass filter

    Returns
    -------
        x: np.ndarray
            FIR filter output signal
        ux: np.ndarray
            point-wise uncertainties associated with x


    References
    ----------
        * Elster and Link 2008 [Elster2008]_

    .. seealso:: :mod:`PyDynamic.deconvolution.fit_filter`


    """
    if not isinstance(sigma_noise, float):
        raise NotImplementedError(
            "FIR formula for covariance propagation not implemented yet. "
            "Suggesting Monte Carlo propagation instead.")

    # Set the FIR filter order.
    Ncomp = len(theta) - 1

    # Handle case of zero uncertainty filter.
    if not isinstance(Utheta, np.ndarray):
        Utheta = np.zeros((Ncomp, Ncomp))

    # Calculate low-pass filtered signal and propagate noise.
    if isinstance(blow, np.ndarray):
        LR = 600
        Bcorr = np.correlate(blow, blow, 'full')
        ycorr = np.convolve(sigma_noise ** 2, Bcorr)
        Lr = len(ycorr)
        Lstart = int(np.ceil(Lr // 2))
        Lend = Lstart + LR - 1
        Ryy = toeplitz(ycorr[Lstart:Lend])
        Ulow = Ryy[:Ncomp + 1, :Ncomp + 1]
        xlow = lfilter(blow, 1.0, y)
    else:
        Ulow = np.eye(len(theta)) * sigma_noise ** 2
        xlow = y

    # Apply FIR filter to calculate best estimate in accordance with GUM.
    x = lfilter(theta, 1.0, xlow)
    x = np.roll(x, -int(shift))

    L = Utheta.shape[0]
    if len(theta.shape) == 1:
        theta = theta[:, np.newaxis]
    # Static part of uncertainty.
    UncCov = theta.T.dot(Ulow.dot(theta)) + np.abs(np.trace(Ulow.dot(Utheta)))
    unc = np.zeros_like(y)
    for m in range(L, len(xlow)):
        # extract necessary part from input signal and apply formula from paper
        XL = xlow[m:m - L:-1, np.newaxis]
        unc[m] = XL.T.dot(Utheta.dot(XL))
    ux = np.sqrt(np.abs(UncCov + unc))
    ux = np.roll(ux, -int(shift))  # correct for delay

    return x, ux.flatten()  # flatten in case that we still have 2D array


def IIRuncFilter(x, noise, b, a, Uab):
    """
    Uncertainty propagation for the signal x and the uncertain IIR filter (b,a)

    Parameters
    ----------
    x: np.ndarray
        filter input signal
    noise: float
        signal noise standard deviation
    b: np.ndarray
        filter numerator coefficients
    a: np.ndarray
        filter denominator coefficients
    Uab: np.ndarray
        covariance matrix for (a[1:],b)

    Returns
    -------
    y: np.ndarray
        filter output signal
    Uy: np.ndarray
        uncertainty associated with y

    References
    ----------
        * Link and Elster [Link2009]_

    """

    if not isinstance(noise, np.ndarray):
        noise = noise * np.ones_like(x)  # translate iid noise to vector

    p = len(a) - 1

    # Adjust dimension for later use.
    if not len(b) == len(a):
        b = np.hstack((b, np.zeros((len(a) - len(b),))))

    # From discrete-time transfer function to state space representation.
    [A, bs, c, b0] = tf2ss(b, a)

    A = np.matrix(A)
    bs = np.matrix(bs)
    c = np.matrix(c)

    phi = np.zeros((2 * p + 1, 1))
    dz = np.zeros((p, p))
    dz1 = np.zeros((p, p))
    z = np.zeros((p, 1))
    P = np.zeros((p, p))

    y = np.zeros((len(x),))
    Uy = np.zeros((len(x),))

    Aabl = np.zeros((p, p, p))
    for k in range(p):
        Aabl[0, k, k] = -1

    # implementation of the state-space formulas from the paper
    for n in range(len(y)):
        for k in range(p):  # derivative w.r.t. a_1,...,a_p
            dz1[:, k] = A * dz[:, k] + np.squeeze(Aabl[:, :, k]) * z
            phi[k] = c * dz[:, k] - b0 * z[k]
        phi[p + 1] = -np.matrix(a[1:]) * z + x[n]  # derivative w.r.t. b_0
        for k in range(p + 2, 2 * p + 1):  # derivative w.r.t. b_1,...,b_p
            phi[k] = z[k - (p + 1)]
        P = A * P * A.T + noise[n] ** 2 * (bs * bs.T)
        y[n] = c * z + b0 * x[n]
        Uy[n] = phi.T * Uab * phi + c * P * c.T + b[0] ** 2 * noise[n] ** 2
        z = A * z + bs * x[n]  # update of the state equations
        dz = dz1

    Uy = np.sqrt(np.abs(Uy))  # calculate point-wise standard uncertainties

    return y, Uy
