# -*- coding: utf-8 -*-
"""
The module :mod:`PyDynamic.model_estimation.fit_filter` contains several functions to
carry out a least-squares fit to a given complex frequency response and the design of
digital deconvolution filters by least-squares fitting to the reciprocal of a given
frequency response each with associated uncertainties.

This module contains the following functions:

* :func:`LSIIR`: Least-squares (time-discrete) IIR filter fit to a given frequency
  response or its reciprocal optionally propagating uncertainties.
* :func:`LSFIR`: Least-squares fit of a digital FIR filter to a given frequency
  response.
* :func:`invLSFIR`: Least-squares fit of a digital FIR filter to the reciprocal of a
  given frequency response.
* :func:`invLSFIR_unc`: Design of FIR filter as fit to reciprocal of frequency response
  values with uncertainty
* :func:`invLSFIR_uncMC`: Design of FIR filter as fit to reciprocal of frequency
  response values with uncertainty via Monte Carlo

"""
import inspect
from typing import Optional, Tuple, Union

import numpy as np
import scipy.signal as dsp
from scipy.optimize import lsq_linear

from ..misc.filterstuff import grpdelay, isstable, mapinside

__all__ = ["LSIIR", "LSFIR", "invLSFIR", "invLSFIR_unc", "invLSFIR_uncMC"]


def _fitIIR(
    Hvals: np.ndarray,
    tau: int,
    w: np.ndarray,
    E: np.ndarray,
    Na: int,
    Nb: int,
    inv: bool = False,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""The actual fitting routing for the least-squares IIR filter.

    Parameters
    ----------
    Hvals : np.ndarray of shape (M,)
        (Complex) frequency response values. If inv is True, then Hvals must not be
        constant zero.
    tau : integer
        initial estimate of time delay
    w : np.ndarray
        :math:`2 * \pi * f / Fs`
    E : np.ndarray
        :math:`exp(-1j * np.dot(w[:, np.newaxis], Ns.T))`
    Nb : int
        numerator polynomial order
    Na : int
        denominator polynomial order
    inv : bool, optional
        If True the least-squares fitting is performed for the reciprocal,
        which means Hvals must not be constant zero then. If False (default) for the
        actual frequency response.
    bounds : 2-tuple of array_like of shape (Na+Nb+1,) or float, optional
         Lower and upper bounds on independent variables. Defaults to no
         bounds. If scalars are provided,the bound will be the same for all
         variables. Use np.inf with an appropriate sign to disable bounds on all or some
         variables. (defaults to unbounded)

    Returns
    -------
    b : np.ndarray
        The IIR filter numerator coefficient vector in a 1-D sequence.
    a : np.ndarray
        The IIR filter denominator coefficient vector in a 1-D sequence.
    """
    if inv and np.all(Hvals == 0):
        raise ValueError(
            f"{inspect.stack()[1].function}: It is not possible to compute the "
            f"reciprocal of zero but the provided frequency "
            f"response{'s are constant ' if len(Hvals) > 1 else 'is '} zero. Please "
            f"provide other frequency responses 'Hvals'."
        )
    exponent = -1 if inv else 1
    Ea = E[:, 1 : Na + 1]
    Eb = E[:, : Nb + 1]
    Htau = np.exp(-1j * w * tau) * Hvals ** exponent
    HEa = np.dot(np.diag(Htau), Ea)
    D = np.hstack((HEa, -Eb))
    Tmp1 = np.real(np.dot(np.conj(D.T), D))
    Tmp2 = np.real(np.dot(np.conj(D.T), -Htau))
    if bounds is None:
        ab = np.linalg.lstsq(Tmp1, Tmp2, rcond=None)[0]
    else:
        ab = lsq_linear(A=Tmp1, b=Tmp2, bounds=bounds).x
    a = np.hstack((1.0, ab[:Na]))
    b = ab[Na:]
    return b, a


def _iterate_stabilization(
    b: np.ndarray,
    a: np.ndarray,
    tau: int,
    w: np.ndarray,
    E: np.ndarray,
    Hvals: np.ndarray,
    Nb: int,
    Na: int,
    Fs: float,
    inv: Optional[bool] = False,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    r"""Conduct one iteration of the stabilization via time delay

    b : np.ndarray
        The initial IIR filter numerator coefficient vector in a 1-D sequence.
    a : np.ndarray
        The initial IIR filter denominator coefficient vector in a 1-D sequence.
    tau : int
        Initial estimate of time delay for filter stabilization.
    w : np.ndarray
        :math:`2 * \pi * f / Fs`
    E : np.ndarray
        :math:`exp(-1j * np.dot(w[:, np.newaxis], Ns.T))`
    Hvals : np.ndarray of shape (M,)
        (complex) frequency response values
    Nb : int
        numerator polynomial order
    Na : int
        denominator polynomial order
    Fs : float
        Sampling frequency for digital IIR filter.
    inv : bool, optional
        If True the least-squares fitting is performed for the reciprocal, if False
        (default) for the actual frequency response
    bounds : 2-tuple of array_like of shape (Na+Nb+1,) or float, optional
         Lower and upper bounds on independent variables. Defaults to no
         bounds. If scalars are provided,the bound will be the same for all
         variables. Use np.inf with an appropriate sign to disable bounds on all or some
         variables. (defaults to unbounded)

    Returns
    -------
    b : np.ndarray
        The IIR filter numerator coefficient vector in a 1-D sequence.
    a : np.ndarray
        The IIR filter denominator coefficient vector in a 1-D sequence.
    tau : int
        Filter time delay (in samples).
    stable : bool
        True if the delayed filter is stable and False if not.
    """
    # Compute appropriate time delay for the stabilization of the filter.
    a_stab = mapinside(a)
    g_1 = grpdelay(b, a, Fs)[0]
    g_2 = grpdelay(b, a_stab, Fs)[0]
    tau += np.ceil(np.median(g_2 - g_1))

    # Conduct stabilization step through time delay.
    b, a = _fitIIR(Hvals, tau, w, E, Na, Nb, inv=inv, bounds=bounds)

    return b, a, tau, isstable(b=b, a=a, ftype="digital")


def LSIIR(
    Hvals: np.ndarray,
    Nb: int,
    Na: int,
    f: np.ndarray,
    Fs: float,
    tau: Optional[int] = 0,
    verbose: Optional[bool] = True,
    max_stab_iter: Optional[int] = 50,
    inv: Optional[bool] = False,
    UHvals: Optional[np.ndarray] = None,
    mc_runs: Optional[int] = 1000,
    bounds: Union[np.ndarray, str] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray, int, np.ndarray]
]:
    """Least-squares (time-discrete) IIR filter fit to frequency response or reciprocal

    For fitting an IIR filter model to the reciprocal of the frequency response values
    or directly to the frequency response values provided by the user, this method
    uses a least-squares fit to determine an estimate of the filter coefficients. The
    filter then optionally is stabilized by pole mapping and introduction of a time
    delay. Associated uncertainties are optionally propagated when provided using the
    GUM S2 Monte Carlo method.

    Parameters
    ----------
    Hvals : array_like of shape (M,)
        (Complex) frequency response values.
    Nb : int
        Order of IIR numerator polynomial.
    Na : int
        Order of IIR denominator polynomial.
    f : array_like of shape (M,)
        Frequencies at which `Hvals` is given.
    Fs : float
        Sampling frequency for digital IIR filter.
    tau : int, optional
        Initial estimate of time delay for obtaining a stable filter (default = 0).
    verbose : bool, optional
        If True (default) be more talkative on stdout. Otherwise no output is written
        anywhere.
    max_stab_iter : int, optional
        Maximum count of iterations for stabilizing the resulting filter. If no
        stabilization should be carried out, this parameter can be set to 0 (default =
        50). This parameter replaced the previous `justFit` which was dropped in
        PyDynamic 2.0.0.
    inv : bool, optional
        If False (default) apply the fit to the frequency response values directly,
        otherwise fit to the reciprocal of the frequency response values.
    UHvals : array_like of shape (2M, 2M), optional
        Uncertainties associated with real and imaginary part of H.
    mc_runs : int, optional
        Number of Monte Carlo runs (default = 1000). Only used if uncertainties
        `UHvals` are provided.
    bounds : 2-tuple of array_like of shape (Na+Nb+1,) or float, optional
         Lower and upper bounds on independent variables. Defaults to no
         bounds. If scalars are provided,the bound will be the same for all
         variables. Use np.inf with an appropriate sign to disable bounds on all or some
         variables. (defaults to unbounded)

    Returns
    -------
    b : np.ndarray
        The IIR filter numerator coefficient vector in a 1-D sequence.
    a : np.ndarray
        The IIR filter denominator coefficient vector in a 1-D sequence.
    tau : int
        Filter time delay (in samples).
    Uab : np.ndarray of shape (Nb+Na+1, Nb+Na+1)
        Uncertainties associated with `[a[1:],b]`. Will only be returned if `UHvals`
        was provided.

    References
    ----------
    * Eichstädt et al. 2010 [Eichst2010]_
    * Vuerinckx et al. 1996 [Vuer1996]_

    .. seealso:: :func:`PyDynamic.uncertainty.propagate_filter.IIRuncFilter`
    """
    # Make sure we enter for loop later on exactly once in case no uncertainty
    # propagation is requested.
    if UHvals is None:
        mc_runs = 1

    # Otherwise augment (the reciprocal of) the frequency response with normally
    # distributed noise according to the covariance matrix provided.
    else:
        # Draw real and imaginary parts of frequency response values with white noise.
        Hvals_ri_unc = np.random.multivariate_normal(
            mean=np.hstack((np.real(Hvals), np.imag(Hvals))), cov=UHvals, size=mc_runs
        )
        Hvals = Hvals_ri_unc[:, : len(f)] + 1j * Hvals_ri_unc[:, len(f) :]

    # Let the user know what we are doing in case it is requested.
    if verbose:
        monte_carlo_message = (
            f" Uncertainties of the filter coefficients are "
            f"evaluated using the GUM S2 Monte Carlo method "
            f"with {mc_runs} runs."
        )
        print(
            f"LSIIR: Least-squares fit of an order {max(Nb, Na)} digital IIR filter to"
            f"{' the reciprocal of' if inv else ''} a frequency response "
            f"given by {len(Hvals)} values.{monte_carlo_message if UHvals else ''}"
        )

    # Initialize the warning message in case the final filter will still be unstable.
    warning_unstable = "CAUTION - The algorithm did NOT result in a stable IIR filter!"

    # Prepare frequencies, fitting and stabilization parameters.
    w = 2 * np.pi * f / Fs
    Ns = np.arange(0, max(Nb, Na) + 1)[:, np.newaxis]
    E = np.exp(-1j * np.dot(w[:, np.newaxis], Ns.T))
    as_and_bs = np.empty((mc_runs, Nb + Na + 1))
    taus = np.empty((mc_runs,), dtype=int)
    tau_max = tau
    stab_iters = np.zeros((mc_runs,), dtype=int)
    if tau == 0 and max_stab_iter == 0:
        relevant_filters = np.ones((mc_runs,), dtype=bool)
    else:
        relevant_filters = np.zeros((mc_runs,), dtype=bool)

    # Conduct the Monte Carlo runs or in case we did not have uncertainties execute
    # just once the actual algorithm.
    for mc_run in range(mc_runs):
        # Conduct actual fit.
        b_i, a_i = _fitIIR(Hvals, tau, w, E, Na, Nb, inv=inv, bounds=bounds)

        # Initialize counter which we use to report about required iteration count.
        current_stab_iter = 1

        # Determine if the computed filter already is stable.
        if isstable(b=b_i, a=a_i, ftype="digital"):
            relevant_filters[mc_run] = True
            taus[mc_run] = tau
        else:
            # If the filter by now is unstable we already tried once to stabilize with
            # initial estimate of time delay and we should iterate at least once. So now
            # we try with previously required maximum time delay to obtain stability.
            if tau_max > tau:
                b_i, a_i = _fitIIR(Hvals, tau_max, w, E, Na, Nb, inv=inv, bounds=bounds)
                current_stab_iter += 1

            if isstable(b=b_i, a=a_i, ftype="digital"):
                relevant_filters[mc_run] = True

            # Set the either needed delay for reaching stability or the initial
            # delay to start iterations.
            taus[mc_run] = tau_max

            # Stabilize filter coefficients with a maximum number of iterations.
            while not relevant_filters[mc_run] and current_stab_iter < max_stab_iter:
                # Compute appropriate time delay for the stabilization of the filter.
                (
                    b_i,
                    a_i,
                    taus[mc_run],
                    relevant_filters[mc_run],
                ) = _iterate_stabilization(
                    b=b_i,
                    a=a_i,
                    tau=taus[mc_run],
                    w=w,
                    E=E,
                    Hvals=Hvals,
                    Nb=Nb,
                    Na=Na,
                    Fs=Fs,
                    inv=inv,
                    bounds=bounds,
                )

                current_stab_iter += 1
            else:
                if taus[mc_run] > tau_max:
                    tau_max = taus[mc_run]
                if verbose:
                    sos = np.sum(np.abs((dsp.freqz(b_i, a_i, w)[1] - Hvals) ** 2))
                    print(
                        f"LSIIR: Fitting "
                        f"{'' if UHvals is None else f'for MC run {mc_run} '}"
                        f"finished. Conducted {current_stab_iter} attempts to "
                        f"stabilize filter. "
                        f"{'' if relevant_filters[mc_run] else warning_unstable} "
                        f"Final sum of squares = {sos}"
                    )

        # Finally store stacked filter parameters.
        as_and_bs[mc_run, :] = np.hstack((a_i[1:], b_i))
        stab_iters[mc_run] = current_stab_iter

    # If we actually ran Monte Carlo simulation we compute the resulting filter.
    if mc_runs > 1:
        # If we did not find any stable filter, calculate the final result from all
        # filters.
        if not np.any(relevant_filters):
            relevant_filters = np.ones_like(relevant_filters)
        b_res = np.mean(as_and_bs[relevant_filters, Na:], axis=0)
        a_res = np.hstack(
            (np.array([1.0]), np.mean(as_and_bs[relevant_filters, :Na], axis=0))
        )
        stab_iter_mean = np.mean(stab_iters[relevant_filters])

        final_stab_iter = 1

        # Determine if the resulting filter already is stable and if not stabilize with
        # an initial delay of the previous maximum delay.
        if not isstable(b=b_res, a=a_res, ftype="digital"):
            final_tau = tau_max
            b_res, a_res = _fitIIR(Hvals, final_tau, w, E, Na, Nb, inv=inv, bounds=bounds)
            final_stab_iter += 1

        final_stable = isstable(b=b_res, a=a_res, ftype="digital")

        while not final_stable and final_stab_iter < max_stab_iter:
            # Compute appropriate time delay for the stabilization of the resulting
            # filter.
            (b_res, a_res, final_tau, final_stable,) = _iterate_stabilization(
                b=b_res,
                a=a_res,
                tau=final_tau,
                w=w,
                E=E,
                Hvals=Hvals,
                Nb=Nb,
                Na=Na,
                Fs=Fs,
                inv=inv,
                bounds=bounds,
            )

            final_stab_iter += 1
    else:
        # If we did not conduct Monte Carlo simulation, we just gather final results.
        b_res = b_i
        a_res = a_i
        stab_iter_mean = final_stab_iter = current_stab_iter
        final_stable = relevant_filters[0]
        final_tau = taus[0]

    if verbose:
        if not final_stable:
            print(
                f"LSIIR: {warning_unstable} Maybe try again with a higher value of "
                f"tau or a higher filter order? Least squares fit finished after "
                f"{stab_iter_mean} stabilization iterations "
                f"{f'on average ' if mc_runs > 1 else ''}"
                f"{f'and with {final_stab_iter} for the final filter ' if final_stab_iter != stab_iter_mean else ''}"
                f"(final tau = {final_tau})."
            )

        Hd = dsp.freqz(b_res, a_res, w)[1] * np.exp(1j * w * tau)
        res = np.hstack((np.real(Hd) - np.real(Hvals), np.imag(Hd) - np.imag(Hvals)))
        rms = np.sqrt(np.sum(res ** 2) / len(f))
        print(f"LSIIR: Final rms error = {rms}.\n\n")

    if UHvals:
        Uab = np.cov(as_and_bs, rowvar=False)
        return b_res, a_res, final_tau, Uab
    else:
        return b_res, a_res, final_tau


def LSFIR(H, N, tau, f, Fs, Wt=None):
    """
    Least-squares fit of a digital FIR filter to a given frequency response.

    Parameters
    ----------
        H : (complex) frequency response values of shape (M,)
        N : FIR filter order
        tau : delay of filter
        f : frequencies of shape (M,)
        Fs : sampling frequency of digital filter
        Wt : (optional) vector of weights of shape (M,) or shape (M,M)

    Returns
    -------
        filter coefficients bFIR (ndarray) of shape (N+1,)

    """

    print("\nLeast-squares fit of an order %d digital FIR filter to the" % N)
    print("reciprocal of a frequency response given by %d values.\n" % len(H))

    H = H[:, np.newaxis]

    w = 2 * np.pi * f / Fs
    w = w[:, np.newaxis]

    ords = np.arange(N + 1)[:, np.newaxis]
    ords = ords.T

    E = np.exp(-1j * np.dot(w, ords))

    if Wt is not None:
        if len(np.shape(Wt)) == 2:  # is matrix
            weights = np.diag(Wt)
        else:
            weights = np.eye(len(f)) * Wt
        X = np.vstack([np.real(np.dot(weights, E)), np.imag(np.dot(weights, E))])
    else:
        X = np.vstack([np.real(E), np.imag(E)])

    H = H * np.exp(1j * w * tau)
    iRI = np.vstack([np.real(1.0 / H), np.imag(1.0 / H)])

    bFIR, res = np.linalg.lstsq(X, iRI)[:2]

    if not isinstance(res, np.ndarray):
        print(
            "Calculation of FIR filter coefficients finished with residual "
            "norm %e" % res
        )

    return np.reshape(bFIR, (N + 1,))


def invLSFIR(H, N, tau, f, Fs, Wt=None):
    """Least-squares fit of a digital FIR filter to the reciprocal of a given
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
        X = np.vstack([np.real(np.dot(weights, E)), np.imag(np.dot(weights, E))])
    else:
        X = np.vstack([np.real(E), np.imag(E)])

    Hs = H * np.exp(1j * w * tau)  # apply time delay for improved fit quality
    iRI = np.vstack([np.real(1.0 / Hs), np.imag(1.0 / Hs)])

    bFIR, res = np.linalg.lstsq(X, iRI)[:2]  # the actual fitting

    if (not isinstance(res, np.ndarray)) or (len(res) == 1):  # summarise results
        print(
            "Calculation of FIR filter coefficients finished with residual "
            "norm %e" % res
        )
        Hd = dsp.freqz(bFIR, 1, 2 * np.pi * f / Fs)[1]
        Hd = Hd * np.exp(1j * 2 * np.pi * f / Fs * tau)
        res = np.hstack((np.real(Hd) - np.real(H), np.imag(Hd) - np.imag(H)))
        rms = np.sqrt(np.sum(res ** 2) / len(f))
        print("Final rms error = %e \n\n" % rms)

    return bFIR.flatten()


def invLSFIR_unc(H, UH, N, tau, f, Fs, wt=None, verbose=True, trunc_svd_tol=None):
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
            array of weights for a weighted least-squares method (default = None
            results in no weighting)
        verbose: bool, optional
            whether to print statements to the command line (default = True)
        trunc_svd_tol: float, optional
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
        print("\nLeast-squares fit of an order %d digital FIR filter to the" % N)
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
    HRI = np.random.multivariate_normal(RI, UH, runs)  # random draws of real,imag of
    # freq response values
    omtau = 2 * np.pi * f / Fs * tau

    # Vectorized Monte Carlo for propagation to inverse
    absHMC = HRI[:, :Nf] ** 2 + HRI[:, Nf:] ** 2
    HiMC = np.hstack(
        (
            (
                HRI[:, :Nf] * np.tile(np.cos(omtau), (runs, 1))
                + HRI[:, Nf:] * np.tile(np.sin(omtau), (runs, 1))
            )
            / absHMC,
            (
                HRI[:, Nf:] * np.tile(np.cos(omtau), (runs, 1))
                - HRI[:, :Nf] * np.tile(np.sin(omtau), (runs, 1))
            )
            / absHMC,
        )
    )
    UiH = np.cov(HiMC, rowvar=False)

    # Step 2: Fit filter coefficients and evaluate uncertainties
    if isinstance(wt, np.ndarray):
        if wt.shape != np.diag(UiH).shape[0]:
            raise ValueError(
                "invLSFIR_unc: User-defined weighting has wrong "
                "dimension. wt is expected to be of length "
                f"{2 * Nf} but is of length {wt.shape}."
            )
    else:
        wt = np.ones(2 * Nf)

    E = np.exp(
        -1j
        * 2
        * np.pi
        * np.dot(f[:, np.newaxis] / Fs, np.arange(N + 1)[:, np.newaxis].T)
    )
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


def invLSFIR_uncMC(H, UH, N, tau, f, Fs, wt=None, verbose=True):
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
        wt: np.ndarray of shape (2M,) - optional
            array of weights for a weighted least-squares method (default = None
            results in no weighting)
        verbose: bool, optional
            whether to print statements to the command line (default = True)

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
        print("\nLeast-squares fit of an order %d digital FIR filter to the" % N)
        print("reciprocal of a frequency response given by %d values" % len(H))
        print("and propagation of associated uncertainties.")

    # Step 1: Propagation of uncertainties to reciprocal of frequency response
    runs = 10000
    HRI = np.random.multivariate_normal(np.hstack((np.real(H), np.imag(H))), UH, runs)

    # Step 2: Fitting the filter coefficients
    Nf = len(f)
    if isinstance(wt, np.ndarray):
        if wt.shape != 2 * Nf:
            raise ValueError(
                "invLSFIR_uncMC: User-defined weighting has wrong "
                "dimension. wt is expected to be of length "
                f"{2 * Nf} but is of length {wt.shape}."
            )
    else:
        wt = np.ones(2 * Nf)

    E = np.exp(
        -1j
        * 2
        * np.pi
        * np.dot(f[:, np.newaxis] / Fs, np.arange(N + 1)[:, np.newaxis].T)
    )
    X = np.vstack((np.real(E), np.imag(E)))
    X = np.dot(np.diag(wt), X)
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


def invLSIIR(Hvals, Nb, Na, f, Fs, tau, justFit=False, verbose=True):
    """Least-squares IIR filter fit to the reciprocal of given frequency response values

    Least-squares fit of a digital IIR filter to the reciprocal of a given set
    of frequency response values and stabilization by pole mapping and introduction
    of a time delay.

    Parameters
    ----------
    Hvals : array_like of shape (M,)
        (Complex) frequency response values.
    Nb : int
        Order of IIR numerator polynomial.
    Na : int
        Order of IIR denominator polynomial.
    f : array_like of shape (M,)
        Frequencies at which `Hvals` is given.
    Fs : float
        Sampling frequency for digital IIR filter.
    tau : int, optional
        Initial estimate of time delay for filter stabilization (default = 0). If
        `justFit = True` this parameter is not used and `tau = 0` will be returned.
    justFit : bool, optional
        If True then no stabilization is carried out, if False (default) filter is
        stabilized.
    verbose : bool, optional
        If True (default) be more talkative on stdout. Otherwise no output is written
        anywhere.

    Returns
    -------
    b : array_like
        The IIR filter numerator coefficient vector in a 1-D sequence.
    a : array_like
        The IIR filter denominator coefficient vector in a 1-D sequence.
    tau : int
        Filter time delay (in samples).

    References
    ----------
    * Eichstädt, Elster, Esward, Hessling [Eichst2010]_

    """
    if justFit:
        return LSIIR(
            Hvals=Hvals,
            Nb=Nb,
            Na=Na,
            f=f,
            Fs=Fs,
            tau=tau,
            verbose=verbose,
            max_stab_iter=0,
            inv=True,
        )
    return LSIIR(
        Hvals=Hvals, Nb=Nb, Na=Na, f=f, Fs=Fs, tau=tau, verbose=verbose, inv=True
    )


def invLSIIR_unc(
    H: np.ndarray,
    UH: np.ndarray,
    Nb: int,
    Na: int,
    f: np.ndarray,
    Fs: float,
    tau: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
    """Stable IIR filter as fit to reciprocal of frequency response with uncertainty

    Least-squares fit of a digital IIR filter to the reciprocal of a given set
    of frequency response values with given associated uncertainty.
    Propagation of uncertainties is carried out using the GUM S2 Monte Carlo method.

    Parameters
    ----------
    H : np.ndarray of shape (M,) and dtype complex
        frequency response values.
    UH : np.ndarray of shape (2M,2M)
        uncertainties associated with real and imaginary part of H
    Nb : int
        order of IIR numerator polynomial.
    Na : int
        order of IIR denominator polynomial.
    f : np.ndarray of shape (M,)
        frequencies corresponding to H
    Fs : float
        sampling frequency for digital IIR filter.
    tau : int
        initial estimate of time delay for filter stabilization.

    Returns
    -------
    b, a : np.ndarray
        IIR filter coefficients
    tau : int
        time delay (in samples)
    Uba : np.ndarray of shape (Nb+Na+1, Nb+Na+1)
        uncertainties associated with [a[1:],b]

    References
    ----------
    * Eichstädt, Elster, Esward and Hessling [Eichst2010]_

    .. seealso:: :mod:`PyDynamic.uncertainty.propagate_filter.IIRuncFilter`
                 :mod:`PyDynamic.model_estimation.fit_filter.invLSIIR`
    """
    print(
        f"invLSIIR_unc: Least-squares fit of an order {max(Nb, Na)} digital IIR "
        f"filter to the reciprocal of a frequency response given by {len(H)} "
        f"values. Uncertainties of the filter coefficients are evaluated using "
        "the GUM S2 Monte Carlo method with 1000 runs."
    )
    return LSIIR(
        Hvals=H,
        Nb=Nb,
        Na=Na,
        f=f,
        Fs=Fs,
        tau=tau,
        verbose=False,
        inv=True,
        UHvals=UH,
        mc_runs=1000,
    )
