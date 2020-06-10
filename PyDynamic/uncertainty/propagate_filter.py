# -*- coding: utf-8 -*-
"""
This module contains functions for the propagation of uncertainties through
the application of a digital filter using the GUM approach.

This modules contains the following functions:

* :func:`FIRuncFilter`: Uncertainty propagation for signal y and uncertain FIR
  filter theta
* :func:`IIRuncFilter`: Uncertainty propagation for the signal x and the uncertain
  IIR filter (b,a)

.. note:: The Elster-Link paper for FIR filters assumes that the autocovariance
          is known and that noise is stationary!

"""

import numpy as np
from scipy.linalg import toeplitz, solve, solve_discrete_lyapunov
from scipy.signal import lfilter, lfilter_zi, dimpulse
from ..misc.tools import trimOrPad

__all__ = ["FIRuncFilter", "IIRuncFilter"]


def FIRuncFilter(y, sigma_noise, theta, Utheta=None, shift=0, blow=None, kind="corr"):
    """Uncertainty propagation for signal y and uncertain FIR filter theta

    Parameters
    ----------
        y: np.ndarray
            filter input signal
        sigma_noise: float or np.ndarray
            float:    standard deviation of white noise in y
            1D-array: interpretation depends on kind
        theta: np.ndarray
            FIR filter coefficients
        Utheta: np.ndarray
            covariance matrix associated with theta
        shift: int
            time delay of filter output signal (in samples)
        blow: np.ndarray
            optional FIR low-pass filter
        kind: string
            defines the interpretation of sigma_noise, if sigma_noise is a 1D-array
            "diag": point-wise standard uncertainties of non-stationary white noise
            "corr": single sided autocovariance of stationary (colored/corrlated) noise (default)

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

    Ntheta = len(theta)  # FIR filter size
    # filterOrder = Ntheta - 1   # FIR filter order

    if not isinstance(Utheta, np.ndarray):  # handle case of zero uncertainty filter
        Utheta = np.zeros((Ntheta, Ntheta))

    # check which case of sigma_noise is necessary
    if isinstance(sigma_noise, float):
        sigma2 = sigma_noise ** 2

    elif isinstance(sigma_noise, np.ndarray):
        if kind == "diag":
            sigma2 = sigma_noise ** 2
        elif kind == "corr":
            sigma2 = sigma_noise
        else:
            raise ValueError("unknown kind of sigma_noise")

    else:
        raise ValueError("sigma_noise is neither of type float nor numpy.ndarray.")


    if isinstance(blow,np.ndarray):             # calculate low-pass filtered signal and propagate noise

        if isinstance(sigma2, float):
            Bcorr = np.correlate(blow, blow, 'full') # len(Bcorr) == 2*Ntheta - 1
            ycorr = sigma2 * Bcorr[len(blow)-1:]     # only the upper half of the correlation is needed

            # trim / pad to length Ntheta
            ycorr = trimOrPad(ycorr, Ntheta)
            Ulow = toeplitz(ycorr)

        elif isinstance(sigma2, np.ndarray):

            if kind == "diag":
                # [Leeuw1994](Covariance matrix of ARMA errors in closed form) can be used, to derive this formula
                # The given "blow" corresponds to a MA(q)-process.
                # Going through the calculations of Leeuw, but assuming
                # that E(vv^T) is a diagonal matrix with non-identical elements,
                # the covariance matrix V becomes (see Leeuw:corollary1)
                # V = N * SP * N^T + M * S * M^T
                # N, M are defined as in the paper
                # and SP is the covariance of input-noise prior to the observed time-interval
                # (SP needs be available len(blow)-timesteps into the past. Here it is
                # assumed, that SP is constant with the first value of sigma2)

                # V needs to be extended to cover Ntheta-1 timesteps more into the past
                sigma2_extended = np.append(sigma2[0] * np.ones((Ntheta - 1)), sigma2)

                N = toeplitz(blow[1:][::-1], np.zeros_like(sigma2_extended)).T
                M = toeplitz(trimOrPad(blow, len(sigma2_extended)), np.zeros_like(sigma2_extended))
                SP = np.diag(sigma2[0] * np.ones_like(blow[1:]))
                S = np.diag(sigma2_extended)

                # Ulow is to be sliced from V, see below
                V = N.dot(SP).dot(N.T) + M.dot(S).dot(M.T)

            elif kind == "corr":

                # adjust the lengths sigma2 to fit blow and theta
                # this either crops (unused) information or appends zero-information
                # note1: this is the reason, why Ulow will have dimension (Ntheta x Ntheta) without further ado

                # calculate Bcorr
                Bcorr = np.correlate(blow, blow, "full")

                # pad or crop length of sigma2, then reflect some part to the left and invert the order
                # [0 1 2 3 4 5 6 7] --> [0 0 0 7 6 5 4 3 2 1 0 1 2 3]
                sigma2 = trimOrPad(sigma2, len(blow) + Ntheta - 1)
                sigma2_reflect = np.pad(sigma2, (len(blow) - 1, 0), mode="reflect")

                ycorr = np.correlate(sigma2_reflect, Bcorr, mode="valid") # used convolve in a earlier version, should make no difference as Bcorr is symmetric
                Ulow = toeplitz(ycorr)

        xlow, _ = lfilter(blow, 1.0, y, zi=y[0] * lfilter_zi(blow, 1.0))

    else:  # if blow is not provided
        if isinstance(sigma2, float):
            Ulow = np.eye(Ntheta) * sigma2

        elif isinstance(sigma2, np.ndarray):

            if kind == "diag":
                # V needs to be extended to cover Ntheta timesteps more into the past
                sigma2_extended = np.append(sigma2[0] * np.ones((Ntheta - 1)), sigma2)

                # Ulow is to be sliced from V, see below
                V = np.diag(sigma2_extended) #  this is not Ulow, same thing as in the case of a provided blow (see above)

            elif kind == "corr":
                Ulow = toeplitz(trimOrPad(sigma2, Ntheta))

        xlow = y

    # apply FIR filter to calculate best estimate in accordance with GUM
    x, _ = lfilter(theta, 1.0, xlow, zi=xlow[0] * lfilter_zi(theta, 1.0))
    x = np.roll(x, -int(shift))

    # add dimension to theta, otherwise transpose won't work
    if len(theta.shape) == 1:
        theta = theta[:, np.newaxis]

    # handle diag-case, where Ulow needs to be sliced from V
    if kind == "diag":
        # UncCov needs to be calculated inside in its own for-loop
        # V has dimension (len(sigma2) + Ntheta) * (len(sigma2) + Ntheta) --> slice a fitting Ulow of dimension (Ntheta x Ntheta)
        UncCov = np.zeros((len(sigma2)))

        for k in range(len(sigma2)):
            Ulow = V[k:k+Ntheta,k:k+Ntheta]
            UncCov[k] = np.squeeze(theta.T.dot(Ulow.dot(theta)) + np.abs(np.trace(Ulow.dot(Utheta))))  # static part of uncertainty

    else:
        UncCov = theta.T.dot(Ulow.dot(theta)) + np.abs(np.trace(Ulow.dot(Utheta)))      # static part of uncertainty

    unc = np.zeros_like(y)
    for m in range(Ntheta,len(xlow)):
        XL = xlow[m:m-Ntheta:-1, np.newaxis]  # extract necessary part from input signal
        unc[m] = XL.T.dot(Utheta.dot(XL))     # apply formula from paper
    ux = np.sqrt(np.abs(UncCov + unc))
    ux = np.roll(ux, -int(shift))  # correct for delay

    return x, ux.flatten()  # flatten in case that we still have 2D array


def IIRuncFilter(x, Ux, b, a, Uab, state = None, kind="diag"):
    """
    Uncertainty propagation for the signal x and the uncertain IIR filter (b,a)

    Parameters
    ----------
        x: np.ndarray
            filter input signal
        Ux: float or np.ndarray
            float:    standard deviation of white noise in y
            1D-array: interpretation depends on kind
        b: np.ndarray
            filter numerator coefficients
        a: np.ndarray
            filter denominator coefficients
        Uab: np.ndarray
            covariance matrix for (a[1:],b)
        kind: string
            defines the interpretation of Ux, if Ux is a 1D-array
            "diag": point-wise standard uncertainties of non-stationary white noise (default)
            "corr": single sided autocovariance of stationary (colored/corrlated) noise
        state: dict
            An internal state (z, dz, P, cache) to start from - e.g. from a previous run of IIRuncFilter.

            * If not given, (z, dz, P) are calculated such that the signal was constant before the given range
            * If given, the input parameters (b, a, Uab) are ignored to avoid repetitive rebuild of the
              internal system description (instead, the cache is used). However a valid new state (i.e.
              with new b, a, Uab) can always be generated by using :py:func:`get_initial_state()`.

    Returns
    -------
        y: np.ndarray
            filter output signal
        Uy: np.ndarray
            uncertainty associated with y
        state: dict
            dictionary of updated internal state

    Note
    ----
        In case of `a == [1.0]` (FIR filter), the results of :py:func:`IIRuncFilter` and :py:func:`FIRuncFilter` will differ!
        This is because IIRuncFilter propagates uncertainty according to the 
        (first-order Taylor series of the) GUM, wheras FIRuncFilter takes full 
        variance information into account (which leads to an additonal term). 
        This is documented in the description of formula (33) of [Elster2008]_ .
        The difference can be visualized by running `examples/Digital filtering/validate_FIR_IIR_MC.py`

    References
    ----------
        * Link and Elster [Link2009]_

    """

    # make Ux an array
    if not isinstance(Ux, np.ndarray):
        Ux = Ux * np.ones_like(x)  # translate iid noise to vector

    # system, corr_unc and processed_input are cached as well to reduce computational load
    if not state:
        # calculate initial state
        if kind == "diag":
            state = get_initial_state(b, a, Uab=Uab, x0=x[0], U0=Ux[0])
        else:  # "corr"
            state = get_initial_state(b, a, Uab=Uab, x0=x[0], U0=Ux[0], Ux=Ux)

    z = state["z"]
    dz = state["dz"]
    P = state["P"]
    A, bs, cT, b0 = state["cache"]["system"]
    corr_unc = state["cache"]["corr_unc"]
    b, a, Uab = state["cache"]["processed_input"]
    p = len(a) - 1

    # phi: dy/dtheta
    phi = np.empty((2 * p + 1, 1))

    # output y, output uncertainty Uy
    y = np.zeros_like(x)
    Uy = np.zeros_like(x)

    # implementation of the state-space formulas from the paper
    for n in range(len(y)):

        # calculate phi according to formulas (13) and (15) from paper
        phi[:p] = np.transpose(cT @ dz - np.transpose(b0 * z[::-1]))   # derivative w.r.t. a_1,...,a_p
        phi[p] = np.dot(-a[1:][::-1], z) + x[n]                        # derivative w.r.t. b_0
        phi[p+1:] = z[::-1]                                            # derivative w.r.t. b_1,...,b_p
        
        # calculate output and output uncertainty according to formulas (7), (12) and (19)
        y[n] = np.dot(cT, z) + b0 * x[n]                                               # (7)
        if isinstance(Uab, np.ndarray):
            if kind == "diag":
                Uy[n] = phi.T @ Uab @ phi + cT @ P @ cT.T + np.square(b0 * Ux[n])      # (12)
            else:  # "corr"
                Uy[n] = phi.T @ Uab @ phi + corr_unc                                   # (19)
        else:  # Uab is None
            if kind == "diag":
                Uy[n] = cT @ P @ cT.T + np.square(b0 * Ux[n])                          # (12)
            else:  # "corr"
                Uy[n] = corr_unc                                                       # (19)

        # timestep update
        if kind == "diag":
            P = A @ P @ A.T + np.square(Ux[n]) * np.outer(bs, bs)   # state uncertainty, formula (18)
        else:  # "corr"
            P = A @ P @ A.T + Ux[0] * np.outer(bs, bs)              # state uncertainty, adopted from formula (18)
        
        #dA_z = np.hstack(dA @ z)                                   # not efficient (dA is sparse), dA = _get_derivative_A(p)
        dA_z = np.vstack((np.zeros((p-1, p)), -z[::-1].T))          # efficient, no tensor-multiplication involved
        dz = A @ dz + dA_z                                          # state derivative, formula (17)
        z = A @ z + bs * x[n]                                       # state, formula (6)
        
    Uy = np.sqrt(np.abs(Uy))  # calculate point-wise standard uncertainties

    # return result and internal state
    state.update({"z": z, "dz": dz, "P": P})
    return y, Uy, state


def _tf2ss(b, a):
    """
    Variant of :py:func:`scipy.signal.tf2ss` that fits the definitions of [Link2009]_ 
    """

    p = len(a) - 1
    A = np.vstack([np.eye(p-1, p, k=1), -a[1:][::-1]])
    B = np.zeros((p, 1))
    B[-1] = 1
    C = np.expand_dims((b[1:] - b[0] * a[1:])[::-1], axis=0)
    D = np.ones((1,1))*b[0]

    return A, B, C, D


def _get_derivative_A(size_A):
    "build tensor representing dA/dtheta"
    dA = np.zeros((size_A, size_A, size_A))
    for k in range(size_A):
        dA[k, -1, -(k+1)] = -1

    return dA


def _get_corr_unc(b, a, Ux):
    """
    Calculate the cumulated correlated noise based on equations (20) of [Link2009]_ .
    """

    # get impulse response of IIR defined by (b,a)
    h_theta = np.squeeze(dimpulse((b, a, 1), x0 = 0.0, t=np.arange(0, len(Ux), step=1))[1][0])
    
    # equation (20), note:
    # - for values r<0 or s<0 the contribution to the sum is zero (because h_theta is zero)
    # - Ux is the one-sided autocorrelation and assumed to be zero outside its range
    corr_unc = np.sum(toeplitz(Ux) * np.outer(h_theta, h_theta))

    return corr_unc


def get_initial_state(b, a, Uab=None, x0 = 1.0, U0 = 1.0, Ux = None):
    """
    Calculate the internal state for the IIRuncFilter-function corresponding to stationary
    non-zero input signal.

    Parameters
    ----------
        b: np.ndarray
            filter numerator coefficients
        a: np.ndarray
            filter denominator coefficients
        x0: float
            stationary input value
        U0: float
            stationary input uncertainty

    Returns
    -------
    internal_state: dict
        dictionary of state
 
    """

    # adjust filter coefficients for lengths
    b, a, Uab = adjust_filter_coefficients(b, a, Uab)

    # convert into state space representation
    [A, B, C, D] = _tf2ss(b, a)

    # necessary intermediate variables
    p = len(A)
    IminusA = np.eye(p) - A
    dA = _get_derivative_A(p)

    # stationary internal state
    # (eye()-A) * zs = B*x0
    zs = solve(IminusA, B) * x0

    # stationary derivative of internal state
    # (eye() - A) dzs = dA * zs
    dzs = solve(IminusA, np.hstack(dA@zs))

    # stationary uncertainty of internal state
    Ps = solve_discrete_lyapunov(A, U0**2 * np.outer(B,B))

    if isinstance(Ux, np.ndarray):
        corr_unc = _get_corr_unc(b, a, Ux)
    else:
        corr_unc = 0

    # bring results into the format that is used within IIRuncFilter
    cache = {"system": (A, B, C, D), "corr_unc": corr_unc, "processed_input": (b, a, Uab)}
    state = {"z": zs, "dz": dzs, "P": Ps, "cache": cache}

    return state


def adjust_filter_coefficients(b, a, Uab):
    """
    Bring b, a to the same length and adjust Uab accordingly
    """
    # length difference of both filters
    d = len(a) - len(b)

    # adjust filter coefficient uncertainties
    if isinstance(Uab, np.ndarray):
        if d != 0:
            if Uab.shape == (len(a)+len(b)-1, len(a)+len(b)-1):
                if d < 0:
                    Uab = np.insert(np.insert(Uab, [len(a)-1]*(-d), 0, axis=0), [len(a)-1]*(-d), 0, axis=1)
                elif d > 0:
                    Uab = np.hstack((Uab, np.zeros((Uab.shape[0], d))))
                    Uab = np.vstack((Uab, np.zeros((d, Uab.shape[1]))))
            else:
                raise ValueError("Uab is not of correct shape.")
    
    # adjust filter coefficients for later use
    if d < 0:
        a = np.hstack((a, np.zeros(-d)))
    elif d > 0:
        b = np.hstack((b, np.zeros(d)))

    return b, a, Uab