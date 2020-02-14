# -*- coding: utf-8 -*-
"""

This module contains functions to carry out a least-squares fit of a digital
filter to the reciprocal of a given
complex frequency response.

This module contains the following functions:

* *LSFIR*: Least-squares fit of a digital FIR filter to the reciprocal of a
  given frequency response.
* *LSFIR_unc*: Design of FIR filter as fit to reciprocal of frequency response
  values with uncertainty
* *LSFIR_uncMC*: Design of FIR filter as fit to reciprocal of frequency
  response values with uncertainty via Monte Carlo
* *LSIIR*: Design of a stable IIR filter as fit to reciprocal of frequency
  response values
* *LSIIR_unc*: Design of a stable IIR filter as fit to reciprocal of frequency
  response values with uncertainty

.. deprecated:: 1.2.71
          The module *deconvolution* will be combined with the module
          *identification* and renamed to *model_estimation* in the next
          major release 2.0.0. From then on you should only use the new module
          *model_estimation* instead.
"""

import warnings

import numpy as np
import scipy.signal as dsp

from ..misc.filterstuff import grpdelay, mapinside

__all__ = ['LSFIR', 'LSFIR_unc', 'LSIIR', 'LSIIR_unc', 'LSFIR_uncMC']

warnings.warn(
    "The module *deconvolution* will be combined with the module "
    "*identification* and renamed to *model_estimation* in the "
    "next major release 2.0.0. From then on you should only use "
    "the new module *model_estimation* instead.",
    DeprecationWarning)


def LSFIR(H, N, tau, f, Fs, Wt=None):
    """	Least-squares fit of a digital FIR filter to the reciprocal of a given
    frequency response.

    Parameters
    ----------
        H: np.ndarray of shape (M,) and dtype complex
            frequency response values
        N: int
            FIR filter order
        tau: float
            delay of filter
        f: np.ndarray of shape (M,)
            frequencies
        Fs: float
            sampling frequency of digital filter
        Wt: np.ndarray of shape (M,) - optional
            vector of weights

    Returns
    -------
        bFIR: np.ndarray of shape (N,)
            filter coefficients

    References
    ----------
        * Elster and Link [Elster2008]_

    .. see_also ::mod::`PyDynamic.uncertainty.propagate_filter.FIRuncFilter`

    """

    print("\nLeast-squares fit of an order %d digital FIR filter to the" % N)
    print("reciprocal of a frequency response given by %d values.\n" % len(H))

    H = H[:, np.newaxis]  # extend to matrix-like for simplified algebra

    w = 2 * np.pi * f / Fs  # set up radial frequencies
    w = w[:, np.newaxis]

    ords = np.arange(N + 1)[:, np.newaxis]  # set up design matrix
    ords = ords.T

    E = np.exp(-1j * np.dot(w, ords))

    if Wt is not None:  # set up weighted design matrix if necessary
        if len(np.shape(Wt)) == 2:  # is matrix
            weights = np.diag(Wt)
        else:
            weights = np.eye(len(f)) * Wt
        X = np.vstack(
            [np.real(np.dot(weights, E)), np.imag(np.dot(weights, E))])
    else:
        X = np.vstack([np.real(E), np.imag(E)])

    Hs = H * np.exp(1j * w * tau)  # apply time delay for improved fit quality
    iRI = np.vstack([np.real(1.0 / Hs), np.imag(1.0 / Hs)])

    bFIR, res = np.linalg.lstsq(X, iRI)[:2]  # the actual fitting

    if (not isinstance(res, np.ndarray)) or (
            len(res) == 1):  # summarise results
        print(
            "Calculation of FIR filter coefficients finished with residual "
            "norm %e" % res)
        Hd = dsp.freqz(bFIR, 1, 2 * np.pi * f / Fs)[1]
        Hd = Hd * np.exp(1j * 2 * np.pi * f / Fs * tau)
        res = np.hstack((np.real(Hd) - np.real(H), np.imag(Hd) - np.imag(H)))
        rms = np.sqrt(np.sum(res ** 2) / len(f))
        print("Final rms error = %e \n\n" % rms)

    return bFIR.flatten()


def LSFIR_unc(H, UH, N, tau, f, Fs, wt=None, verbose=True, trunc_svd_tol=None):
    """Design of FIR filter as fit to reciprocal of frequency response values
    with uncertainty


    Least-squares fit of a digital FIR filter to the reciprocal of a
    frequency response
    for which associated uncertainties are given for its real and imaginary
    part.
    Uncertainties are propagated using a truncated svd and linear matrix
    propagation.

    Parameters
    ----------
        H: np.ndarray of shape (M,)
            frequency response values
        UH: np.ndarray of shape (2M,2M)
            uncertainties associated with the real and imaginary part
        N: int
            FIR filter order
        tau: float
            delay of filter
        f: np.ndarray of shape (M,)
            frequencies
        Fs: float
            sampling frequency of digital filter
        wt: np.ndarray of shape (2M,) - optional
            array of weights for a weighted least-squares method
        verbose: bool, optional
            whether to print statements to the command line
        trunc_svd_tol: float
            lower bound for singular values to be considered for pseudo-inverse

    Returns
    -------
        b: np.ndarray of shape (N+1,)
            filter coefficients of shape
        Ub: np.ndarray of shape (N+1,N+1)
            uncertainties associated with b

    References
    ----------
        * Elster and Link [Elster2008]_
    """

    if verbose:
        print(
            "\nLeast-squares fit of an order %d digital FIR filter to the" % N)
        print("reciprocal of a frequency response given by %d values" % len(H))
        print("and propagation of associated uncertainties.")

    # Step 1: Propagation of uncertainties to reciprocal of frequency response
    runs = 10000
    Nf = len(f)

    if not len(H) == UH.shape[0]:
        # Assume that H is given as complex valued frequency response.
        RI = np.hstack((np.real(H), np.imag(H)))
    else:
        RI = H.copy()
        H = H[:Nf] + 1j * H[Nf:]
    HRI = np.random.multivariate_normal(RI, UH,
                                        runs)  # random draws of real,imag of
    # freq response values
    omtau = 2 * np.pi * f / Fs * tau

    # Vectorized Monte Carlo for propagation to inverse
    absHMC = HRI[:, :Nf] ** 2 + HRI[:, Nf:] ** 2
    HiMC = np.hstack(((HRI[:, :Nf] * np.tile(np.cos(omtau), (runs, 1)) +
                       HRI[:, Nf:] * np.tile(np.sin(omtau), (runs, 1))) /
                      absHMC,
                      (HRI[:, Nf:] * np.tile(np.cos(omtau), (runs, 1)) -
                       HRI[:, :Nf] * np.tile(np.sin(omtau), (runs, 1))) /
                      absHMC))
    UiH = np.cov(HiMC, rowvar=False)

    # Step 2: Fit filter coefficients and evaluate uncertainties
    if isinstance(wt, np.ndarray):
        if wt.shape != np.diag(UiH).shape[0]:
            raise ValueError("User-defined weighting has wrong dimension.")
    else:
        wt = np.ones(2 * Nf)

    E = np.exp(-1j * 2 * np.pi * np.dot(f[:, np.newaxis] / Fs,
                                        np.arange(N + 1)[:, np.newaxis].T))
    X = np.vstack((np.real(E), np.imag(E)))
    X = np.dot(np.diag(wt), X)
    Hm = H * np.exp(1j * 2 * np.pi * f / Fs * tau)
    Hri = np.hstack((np.real(1.0 / Hm), np.imag(1.0 / Hm)))

    u, s, v = np.linalg.svd(X, full_matrices=False)
    if isinstance(trunc_svd_tol, float):
        s[s < trunc_svd_tol] = 0.0
    StSInv = np.zeros_like(s)
    StSInv[s > 0] = s[s > 0] ** (-2)

    M = np.dot(np.dot(np.dot(v.T, np.diag(StSInv)), np.diag(s)), u.T)

    bFIR = np.dot(M, Hri[:, np.newaxis])  # actual fitting
    UbFIR = np.dot(np.dot(M, UiH), M.T)  # evaluation of uncertainties

    bFIR = bFIR.flatten()

    if verbose:
        Hd = dsp.freqz(bFIR, 1, 2 * np.pi * f / Fs)[1]
        Hd = Hd * np.exp(1j * 2 * np.pi * f / Fs * tau)
        res = np.hstack((np.real(Hd) - np.real(H), np.imag(Hd) - np.imag(H)))
        rms = np.sqrt(np.sum(res ** 2) / len(f))
        print("Final rms error = %e \n\n" % rms)

    return bFIR, UbFIR


def LSFIR_uncMC(H, UH, N, tau, f, Fs, verbose=True):
    """Design of FIR filter as fit to reciprocal of frequency response values
    with uncertainty

    Least-squares fit of a FIR filter to the reciprocal of a frequency response
    for which associated uncertainties are given for its real and imaginary
    parts.
    Uncertainties are propagated using a Monte Carlo method. This method may
    help in cases where
    the weighting matrix or the Jacobian are ill-conditioned, resulting in
    false uncertainties
    associated with the filter coefficients.

    Parameters
    ----------
        H: np.ndarray of shape (M,) and dtype complex
            frequency response values
        UH: np.ndarray of shape (2M,2M)
            uncertainties associated with the real and imaginary part of H
        N: int
            FIR filter order
        tau: int
            time delay of filter in samples
        f: np.ndarray of shape (M,)
            frequencies corresponding to H
        Fs: float
            sampling frequency of digital filter
        verbose: bool, optional
            whether to print statements to the command line

    Returns
    -------
        b: np.ndarray of shape (N+1,)
            filter coefficients of shape
        Ub: np.ndarray of shape (N+1, N+1)
            uncertainties associated with b

    References
    ----------
        * Elster and Link [Elster2008]_
    """

    if verbose:
        print(
            "\nLeast-squares fit of an order %d digital FIR filter to the" % N)
        print("reciprocal of a frequency response given by %d values" % len(H))
        print("and propagation of associated uncertainties.")

    # Step 1: Propagation of uncertainties to reciprocal of frequency response
    runs = 10000
    HRI = np.random.multivariate_normal(np.hstack((np.real(H), np.imag(H))), UH,
                                        runs)

    # Step 2: Fitting the filter coefficients
    E = np.exp(-1j * 2 * np.pi * np.dot(f[:, np.newaxis] / Fs,
                                        np.arange(N + 1)[:, np.newaxis].T))
    X = np.vstack((np.real(E), np.imag(E)))

    Nf = len(f)
    bF = np.zeros((N + 1, runs))
    resn = np.zeros((runs,))
    for k in range(runs):
        Hk = HRI[k, :Nf] + 1j * HRI[k, Nf:]
        Hkt = Hk * np.exp(1j * 2 * np.pi * f / Fs * tau)
        iRI = np.hstack([np.real(1.0 / Hkt), np.imag(1.0 / Hkt)])
        bF[:, k], res = np.linalg.lstsq(X, iRI)[:2]
        resn[k] = np.linalg.norm(res)

    bFIR = np.mean(bF, axis=1)
    UbFIR = np.cov(bF, rowvar=True)

    return bFIR, UbFIR


def fitIIR(Hvals, tau, w, E, Na, Nb):
    """The helper function for the actual fitting part.

    Parameters
    ----------
        Hvals:   np.ndarray of shape (M,) and dtype complex
            frequency response values.
        tau: float
            initial estimate of time delay for filter stabilization.
        w:       np.ndarray
            :math:`2 * np.pi * f / Fs`
        E:       np.ndarray
            :math:`np.exp(-1j * np.dot(w[:, np.newaxis], Ns.T))`
        Nb: int
            order of IIR numerator polynomial.
        Na: int
            order of IIR denominator polynomial.

    Returns
    -------
        bi, ai : np.ndarray
            IIR filter coefficients as numpy arrays
    """
    from numpy import conj
    from numpy.linalg import lstsq

    # helper function for actual fitting part
    Ea = E[:, 1:Na + 1]
    Eb = E[:, :Nb + 1]
    Htau = np.exp(-1j * w * tau) * Hvals ** (-1)
    HEa = np.dot(np.diag(Htau), Ea)
    D = np.hstack((HEa, -Eb))
    Tmp1 = np.real(np.dot(conj(D.T), D))
    Tmp2 = np.real(np.dot(conj(D.T), -Htau))
    ab = lstsq(Tmp1, Tmp2)[0]
    ai = np.hstack((1.0, ab[:Na]))
    bi = ab[Na:]
    return bi, ai


def LSIIR(Hvals, Nb, Na, f, Fs, tau, justFit=False, verbose=True):
    """Design of a stable IIR filter as fit to reciprocal of frequency
    response values

    Least-squares fit of a digital IIR filter to the reciprocal of a given set
    of frequency response values using the equation-error method and
    stabilization
    by pole mapping and introduction of a time delay.

    Parameters
    ----------
        Hvals: np.ndarray of shape (M,) and dtype complex
            frequency response values.
        Nb: int
            order of IIR numerator polynomial.
        Na: int
            order of IIR denominator polynomial.
        f: np.ndarray of shape (M,)
            frequencies corresponding to Hvals
        Fs: float
            sampling frequency for digital IIR filter.
        tau: float
            initial estimate of time delay for filter stabilization.
        justFit: bool
            if True then no stabilization is carried out.
        verbose: bool
            If True print some more detail about input parameters.

    Returns
    -------
        b, a : np.ndarray
            IIR filter coefficients
        tau : int
            time delay (in samples)

    References
    ----------
        * Eichstädt, Elster, Esward, Hessling [Eichst2010]_

    """
    from numpy import count_nonzero, roots, ceil, median

    if verbose:
        print(
            "\nLeast-squares fit of an order %d digital IIR filter to the" %
            max(
                Nb, Na))
        print("reciprocal of a frequency response given by %d values.\n" % len(
            Hvals))

    w = 2 * np.pi * f / Fs
    Ns = np.arange(0, max(Nb, Na) + 1)[:, np.newaxis]
    E = np.exp(-1j * np.dot(w[:, np.newaxis], Ns.T))

    bi, ai = fitIIR(Hvals, tau, w, E, Na, Nb)

    if justFit:  # no uncertainty evaluation
        return bi, ai

    if count_nonzero(abs(roots(ai)) > 1) > 0:
        stable = False
    else:
        stable = True

    maxiter = 50

    astab = mapinside(ai)  # stabilise filter
    run = 1

    while stable is not True and run < maxiter:  # shift delay such that filter
        # becomes stable
        g1 = grpdelay(bi, ai, Fs)[0]
        g2 = grpdelay(bi, astab, Fs)[0]
        tau = ceil(tau + median(g2 - g1))

        bi, ai = fitIIR(Hvals, tau, w, E, Na, Nb)
        if count_nonzero(abs(roots(ai)) > 1) > 0:
            astab = mapinside(ai)
        else:
            stable = True
        run = run + 1

    if count_nonzero(abs(roots(ai)) > 1) > 0 and verbose:
        print("Caution: The algorithm did NOT result in a stable IIR filter!")
        print(
            "Maybe try again with a higher value of tau0 or a higher filter "
            "order?")

    if verbose:
        print("Least squares fit finished after %d iterations (tau=%d).\n" % (
            run, tau))
        Hd = dsp.freqz(bi, ai, 2 * np.pi * f / Fs)[1]
        Hd = Hd * np.exp(1j * 2 * np.pi * f / Fs * tau)
        res = np.hstack(
            (np.real(Hd) - np.real(Hvals), np.imag(Hd) - np.imag(Hvals)))
        rms = np.sqrt(np.sum(res ** 2) / len(f))
        print("Final rms error = %e \n\n" % rms)

    return bi, ai, int(tau)


def LSIIR_unc(H, UH, Nb, Na, f, Fs, tau=0):
    """Design of stabel IIR filter as fit to reciprocal of given frequency
    response with uncertainty

    Least-squares fit of a digital IIR filter to the reciprocal of a given set
    of frequency response values with given associated uncertainty.
    Propagation of uncertainties is
    carried out using the Monte Carlo method.

    Parameters
    ----------

        H: np.ndarray of shape (M,) and dtype complex
            frequency response values.
        UH: np.ndarray of shape (2M,2M)
            uncertainties associated with real and imaginary part of H
        Nb: int
            order of IIR numerator polynomial.
        Na: int
            order of IIR denominator polynomial.
        f: np.ndarray of shape (M,)
            frequencies corresponding to H
        Fs: float
            sampling frequency for digital IIR filter.
        tau: float
            initial estimate of time delay for filter stabilization.

    Returns
    -------
        b,a: np.ndarray
            IIR filter coefficients
        tau: int
            time delay (in samples)
        Uba: np.ndarray of shape (Nb+Na+1, Nb+Na+1)
            uncertainties associated with [a[1:],b]

    References
    ----------
        * Eichstädt, Elster, Esward and Hessling [Eichst2010]_

    .. seealso:: :mod:`PyDynamic.uncertainty.propagate_filter.IIRuncFilter`
                 :mod:`PyDynamic.deconvolution.fit_filter.LSIIR`
    """

    runs = 1000

    print(
        "\nLeast-squares fit of an order %d digital IIR filter to the" %
        max(Nb, Na))
    print("reciprocal of a frequency response given by %d values.\n" % len(H))
    print("Uncertainties of the filter coefficients are evaluated using\n"
          "the GUM S2 Monte Carlo method with %d runs.\n" % runs)

    # Step 1: Propagation of uncertainties to frequency response
    HRI = np.random.multivariate_normal(np.hstack((np.real(H), np.imag(H))), UH,
                                        runs)
    HH = HRI[:, :len(f)] + 1j * HRI[:, len(f):]

    # Step 2: Fit filter and evaluate uncertainties (Monte Carlo method)
    AB = np.zeros((runs, Nb + Na + 1))
    Tau = np.zeros((runs,))
    for k in range(runs):
        bi, ai, Tau[k] = LSIIR(HH[k, :], Nb, Na, f, Fs, tau, verbose=False)
        AB[k, :] = np.hstack((ai[1:], bi))

    bi = np.mean(AB[:, Na:], axis=0)
    ai = np.hstack((np.array([1.0]), np.mean(AB[:, :Na], axis=0)))
    Uab = np.cov(AB, rowvar=False)
    tau = np.mean(Tau)
    return bi, ai, tau, Uab
