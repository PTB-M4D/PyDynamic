"""This module assists in carrying out least-squares IIR and FIR filter fits

It is possible to carry out a least-squares fit of digital, time-discrete IIR and FIR
filters to a given complex frequency response and the design of digital deconvolution
filters by least-squares fitting to the reciprocal of a given frequency response each
with propagation of associated uncertainties.

This module contains the following functions:

* :func:`LSIIR`: Least-squares (time-discrete) IIR filter fit to a given frequency
  response or its reciprocal optionally propagating uncertainties.
* :func:`LSFIR`: Least-squares fit of a digital FIR filter to a given frequency
  response.
* :func:`invLSFIR`: Least-squares fit of a digital FIR filter to the reciprocal of a
  given frequency response.
* :func:`invLSFIR_unc`: Design of FIR filter as fit to reciprocal of frequency response
  values with uncertainty propagation via a singular-value decomposition and
  linear matrix propagation.
* :func:`invLSFIR_uncMC`: Least-squares (time-discrete) FIR filter fit to a given
  frequency response or its reciprocal optionally propagating uncertainties either
  via Monte Carlo or via a singular-value decomposition and linear matrix propagation.

"""

__all__ = ["LSIIR", "LSFIR", "invLSFIR", "invLSFIR_unc", "invLSFIR_uncMC"]

import inspect
from enum import Enum
from os import path
from typing import Optional, Tuple, Union

import numpy as np
import scipy.signal as dsp
from scipy.stats import multivariate_normal

from ..misc.filterstuff import grpdelay, isstable, mapinside
from ..misc.tools import (
    is_2d_matrix,
    is_2d_square_matrix,
    number_of_rows_equals_vector_dim,
)


def LSIIR(
    H: np.ndarray,
    Nb: int,
    Na: int,
    f: np.ndarray,
    Fs: float,
    tau: Optional[int] = 0,
    verbose: Optional[bool] = True,
    max_stab_iter: Optional[int] = 50,
    inv: Optional[bool] = False,
    UH: Optional[np.ndarray] = None,
    mc_runs: Optional[int] = 1000,
) -> Tuple[np.ndarray, np.ndarray, int, Union[np.ndarray, None]]:
    """Least-squares (time-discrete) IIR filter fit to frequency response or reciprocal

    For fitting an IIR filter model to the reciprocal of the frequency response values
    or directly to the frequency response values provided by the user, this method
    uses a least-squares fit to determine an estimate of the filter coefficients. The
    filter then optionally is stabilized by pole mapping and introduction of a time
    delay. Associated uncertainties are optionally propagated when provided using the
    GUM S2 Monte Carlo method.

    Parameters
    ----------
    H : array_like of shape (M,)
        (Complex) frequency response values.
    Nb : int
        Order of IIR numerator polynomial.
    Na : int
        Order of IIR denominator polynomial.
    f : array_like of shape (M,)
        Frequencies at which H is given.
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
    UH : array_like of shape (2M, 2M), optional
        Uncertainties associated with real and imaginary part of H.
    mc_runs : int, optional
        Number of Monte Carlo runs (default = 1000). Only used if uncertainties
        UH are provided.

    Returns
    -------
    b : np.ndarray
        The IIR filter numerator coefficient vector in a 1-D sequence.
    a : np.ndarray
        The IIR filter denominator coefficient vector in a 1-D sequence.
    tau : int
        Filter time delay (in samples).
    Uab : np.ndarray of shape (Nb+Na+1, Nb+Na+1)
        Uncertainties associated with `[a[1:],b]`. Will be None if UH is not
        provided or is None.

    References
    ----------
    * Eichstädt et al. 2010 [Eichst2010]_
    * Vuerinckx et al. 1996 [Vuer1996]_

    .. seealso:: :func:`PyDynamic.uncertainty.propagate_filter.IIRuncFilter`
    """
    if _no_uncertainties_were_provided(UH):
        freq_resp_to_fit, mc_runs = H, 1
    else:
        freq_resp_to_fit = _assemble_complex_from_real_imag(
            _draw_multivariate_monte_carlo_samples(
                _assemble_real_imag_from_complex(H), UH, mc_runs
            )
        )

    if verbose:
        _print_iir_welcome_msg(H, Na, Nb, UH, inv, mc_runs)

    warn_unstable_msg = "CAUTION - The algorithm did NOT result in a stable IIR filter!"

    # Prepare frequencies, fitting and stabilization parameters.
    omega = _compute_radial_freqs_equals_two_pi_times_freqs_over_sampling_freq(Fs, f)
    Ns = np.arange(0, max(Nb, Na) + 1)[:, np.newaxis]
    E = np.exp(-1j * np.dot(omega[:, np.newaxis], Ns.T))
    as_and_bs = np.empty((mc_runs, Nb + Na + 1))
    taus = np.empty((mc_runs,), dtype=int)
    tau_max = tau
    stab_iters = np.zeros((mc_runs,), dtype=int)
    if tau == 0 and max_stab_iter == 0:
        relevant_filters_mask = np.ones((mc_runs,), dtype=bool)
    else:
        relevant_filters_mask = np.zeros((mc_runs,), dtype=bool)

    for mc_run in range(mc_runs):
        b_i, a_i = _compute_actual_iir_least_squares_fit(
            freq_resp_to_fit, tau, omega, E, Na, Nb, inv
        )

        current_stabilization_iteration_counter = 1
        if isstable(b_i, a_i, "digital"):
            relevant_filters_mask[mc_run] = True
            taus[mc_run] = tau
        else:
            # If the filter by now is unstable we already tried once to stabilize with
            # initial estimate of time delay and we should iterate at least once. So now
            # we try with previously required maximum time delay to obtain stability.
            if tau_max > tau:
                b_i, a_i = _compute_actual_iir_least_squares_fit(
                    freq_resp_to_fit, tau_max, omega, E, Na, Nb, inv
                )
                current_stabilization_iteration_counter += 1

            if isstable(b_i, a_i, "digital"):
                relevant_filters_mask[mc_run] = True

            # Set the either needed delay for reaching stability or the initial
            # delay to start iterations.
            taus[mc_run] = tau_max

            while (
                not relevant_filters_mask[mc_run]
                and current_stabilization_iteration_counter < max_stab_iter
            ):
                (
                    b_i,
                    a_i,
                    taus[mc_run],
                    relevant_filters_mask[mc_run],
                ) = _compute_stabilized_filter_through_time_delay_iteration(
                    b_i, a_i, taus[mc_run], omega, E, freq_resp_to_fit, Nb, Na, Fs, inv
                )
                current_stabilization_iteration_counter += 1
            else:
                if taus[mc_run] > tau_max:
                    tau_max = taus[mc_run]
                if verbose:
                    sos = np.sum(
                        np.abs((dsp.freqz(b_i, a_i, omega)[1] - freq_resp_to_fit) ** 2)
                    )
                    print(
                        f"LSIIR: Fitting{'' if UH is None else f' for MC run {mc_run}'}"
                        f" finished. Conducted "
                        f"{current_stabilization_iteration_counter} attempts to "
                        f"stabilize filter. "
                        f"{'' if relevant_filters_mask[mc_run] else warn_unstable_msg} "
                        f"Final sum of squares = {sos}"
                    )

        # Finally store stacked filter parameters.
        as_and_bs[mc_run, :] = np.hstack((a_i[1:], b_i))
        stab_iters[mc_run] = current_stabilization_iteration_counter

    # If we actually ran Monte Carlo simulation we compute the resulting filter.
    if mc_runs > 1:
        # If we did not find any stable filter, calculate the final result from all
        # filters.
        if not np.any(relevant_filters_mask):
            relevant_filters_mask = np.ones_like(relevant_filters_mask)
        b_res = np.mean(as_and_bs[relevant_filters_mask, Na:], axis=0)
        a_res = np.hstack(
            (np.array([1.0]), np.mean(as_and_bs[relevant_filters_mask, :Na], axis=0))
        )
        stab_iter_mean = np.mean(stab_iters[relevant_filters_mask])

        final_stabilization_iteration_counter = 1

        # Determine if the resulting filter already is stable and if not stabilize with
        # an initial delay of the previous maximum delay.
        if not isstable(b_res, a_res, "digital"):
            final_tau = tau_max
            b_res, a_res = _compute_actual_iir_least_squares_fit(
                freq_resp_to_fit, final_tau, omega, E, Na, Nb, inv
            )
            final_stabilization_iteration_counter += 1

        final_stable = isstable(b_res, a_res, "digital")

        while (
            not final_stable and final_stabilization_iteration_counter < max_stab_iter
        ):
            # Compute appropriate time delay for the stabilization of the resulting
            # filter.
            (
                b_res,
                a_res,
                final_tau,
                final_stable,
            ) = _compute_stabilized_filter_through_time_delay_iteration(
                b_res, a_res, final_tau, omega, E, freq_resp_to_fit, Nb, Na, Fs, inv
            )

            final_stabilization_iteration_counter += 1
    else:
        # If we did not conduct Monte Carlo simulation, we just gather final results.
        b_res = b_i
        a_res = a_i
        stab_iter_mean = (
            final_stabilization_iteration_counter
        ) = current_stabilization_iteration_counter
        final_stable = relevant_filters_mask[0]
        final_tau = taus[0]

    if verbose:
        if not final_stable:
            final_stabilization_msg = (
                f"and with {final_stabilization_iteration_counter} for the final "
                f"filter "
                if final_stabilization_iteration_counter != stab_iter_mean
                else ""
            )
            print(
                f"LSIIR: {warn_unstable_msg} Maybe try again with a higher value of "
                f"tau or a higher filter order? Least squares fit finished after "
                f"{stab_iter_mean} stabilization iterations "
                f"{f'on average ' if mc_runs > 1 else ''}{final_stabilization_msg}"
                f"(final tau = {final_tau})."
            )
        Hd = _compute_delayed_filters_freq_resp_via_scipys_freqz(
            b_res, a_res, tau, omega
        )
        residuals_real_imag = _assemble_real_imag_from_complex(Hd - freq_resp_to_fit)
        _compute_and_print_rms(residuals_real_imag)

    if UH:
        Uab = np.cov(as_and_bs, rowvar=False)
        return b_res, a_res, final_tau, Uab
    return b_res, a_res, final_tau, None


def _print_iir_welcome_msg(
    H: np.ndarray, Na: int, Nb: int, UH: np.ndarray, inv: bool, mc_runs: int
):
    monte_carlo_message = (
        f" Uncertainties of the filter coefficients are "
        f"evaluated using the GUM S2 Monte Carlo method "
        f"with {mc_runs} runs."
    )
    print(
        f"LSIIR: Least-squares fit of an order {max(Nb, Na)} digital IIR filter to"
        f"{' the reciprocal of' if inv else ''} a frequency response "
        f"given by {len(H)} values.{monte_carlo_message if UH else ''}"
    )


def _compute_radial_freqs_equals_two_pi_times_freqs_over_sampling_freq(
    sampling_freq: float, freqs: np.ndarray
) -> np.ndarray:
    return 2 * np.pi * freqs / sampling_freq


def _compute_actual_iir_least_squares_fit(
    H: np.ndarray,
    tau: int,
    omega: np.ndarray,
    E: np.ndarray,
    Na: int,
    Nb: int,
    inv: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if inv and np.all(H == 0):
        raise ValueError(
            f"{_get_first_public_caller()}: It is not possible to compute the "
            f"reciprocal of zero but the provided frequency response"
            f"{'s are constant ' if len(H) > 1 else 'is '} zero. Please "
            f"provide other frequency responses Hvals."
        )
    Ea = E[:, 1 : Na + 1]
    Eb = E[:, : Nb + 1]
    e_to_the_minus_one_j_omega_tau = _compute_e_to_the_one_j_omega_tau(-omega, tau)
    delayed_freq_resp_or_recipr = e_to_the_minus_one_j_omega_tau * (
        np.reciprocal(H) if inv else H
    )
    HEa = np.dot(np.diag(delayed_freq_resp_or_recipr), Ea)
    D = np.hstack((HEa, -Eb))
    Tmp1 = np.real(np.dot(np.conj(D.T), D))
    Tmp2 = np.real(np.dot(np.conj(D.T), -delayed_freq_resp_or_recipr))
    ab = _fit_filter_coeffs_via_least_squares(Tmp1, Tmp2)
    a = np.hstack((1.0, ab[:Na]))
    b = ab[Na:]
    return b, a


def _compute_stabilized_filter_through_time_delay_iteration(
    b: np.ndarray,
    a: np.ndarray,
    tau: int,
    w: np.ndarray,
    E: np.ndarray,
    H: np.ndarray,
    Nb: int,
    Na: int,
    Fs: float,
    inv: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    tau += _compute_filter_stabilization_time_delay(b, a, Fs)

    b, a = _compute_one_filter_stabilization_iteration_through_time_delay(
        H, tau, w, E, Nb, Na, inv
    )

    return b, a, tau, isstable(b, a, "digital")


def _compute_filter_stabilization_time_delay(
    b: np.ndarray,
    a: np.ndarray,
    Fs: float,
) -> int:
    a_stabilized = mapinside(a)
    g_1 = grpdelay(b, a, Fs)[0]
    g_2 = grpdelay(b, a_stabilized, Fs)[0]
    return np.ceil(np.median(g_2 - g_1))


def _compute_one_filter_stabilization_iteration_through_time_delay(
    H: np.ndarray,
    tau: int,
    w: np.ndarray,
    E: np.ndarray,
    Nb: int,
    Na: int,
    inv: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    return _compute_actual_iir_least_squares_fit(H, tau, w, E, Na, Nb, inv)


def _compute_delayed_filters_freq_resp_via_scipys_freqz(
    b: np.ndarray,
    a: Union[float, np.ndarray],
    tau: int,
    omega: np.ndarray,
) -> np.ndarray:
    filters_freq_resp = dsp.freqz(b, a, omega)[1]
    delayed_filters_freq_resp = filters_freq_resp * _compute_e_to_the_one_j_omega_tau(
        omega, tau
    )
    return delayed_filters_freq_resp


def LSFIR(
    H: np.ndarray,
    N: int,
    tau: int,
    f: np.ndarray,
    Fs: float,
    Wt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Least-squares (time-discrete) digital FIR filter fit to frequency response

    Parameters
    ----------
    H : array_like of shape (M,)
        (Complex) frequency response values
    N : int
        FIR filter order
    tau : int
        delay of filter
    f : array_like of shape (M,)
        Frequencies at which ``H`` is given
    Fs : float
        sampling frequency of digital FIR filter.
    Wt : array_like of shape (M,) or shape (M,M), optional
        vector of weights

    Returns
    -------
    b : np.ndarray of shape (N+1,)
        The FIR filter coefficient vector in a 1-D sequence
    """

    print(
        f"\nLSFIR: Least-squares fit of an order {N} digital FIR filter to the "
        f"frequency response H given by {len(H)} values.\n"
    )

    frequencies = f.copy()
    sampling_frequency = Fs

    n_frequencies = len(frequencies)
    h_complex = _assemble_complex_from_real_imag(H)

    omega = _compute_radial_freqs_equals_two_pi_times_freqs_over_sampling_freq(
        sampling_frequency, frequencies
    )

    weights = _validate_and_return_weights(weights=Wt, expected_len=2 * n_frequencies)

    x = _compute_x(N, frequencies, sampling_frequency, weights)

    delayed_h_complex = h_complex * _compute_e_to_the_one_j_omega_tau(omega, tau)
    iRI = _assemble_real_imag_from_complex(delayed_h_complex)

    bFIR, res = np.linalg.lstsq(x, iRI, rcond=None)[:2]

    if not isinstance(res, np.ndarray):
        print(
            "LSFIR: Calculation of FIR filter coefficients finished with residual "
            f"norm {res}."
        )

    return bFIR


def _assemble_complex_from_real_imag(array: np.ndarray) -> np.ndarray:
    if is_2d_matrix(array):
        array_split_in_two_half = (
            _split_array_of_monte_carlo_samples_of_real_and_imag_parts(array)
        )
    else:
        array_split_in_two_half = _split_vector_of_real_and_imag_parts(array)
    return array_split_in_two_half[0] + 1j * array_split_in_two_half[1]


def _split_array_of_monte_carlo_samples_of_real_and_imag_parts(
    array: np.ndarray,
) -> np.ndarray:
    return np.split(ary=array, indices_or_sections=2, axis=1)


def _split_vector_of_real_and_imag_parts(vector: np.ndarray) -> np.ndarray:
    return np.split(ary=vector, indices_or_sections=2)


def _compute_x(
    filter_order: int,
    freqs: np.ndarray,
    sampling_freq: float,
    weights: np.ndarray,
):
    e = np.exp(
        -1j
        * 2
        * np.pi
        * np.dot(
            freqs[:, np.newaxis] / sampling_freq,
            np.arange(filter_order + 1)[:, np.newaxis].T,
        )
    )
    x = np.vstack((np.real(e), np.imag(e)))
    x = np.dot(np.diag(weights), x)
    return x


def _assemble_real_imag_from_complex(array: np.ndarray) -> np.ndarray:
    return np.hstack((np.real(array), np.imag(array)))


def invLSFIR(
    H: np.ndarray,
    N: int,
    tau: int,
    f: np.ndarray,
    Fs: float,
    Wt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Least-squares (time-discrete) digital FIR filter fit to freq. resp. reciprocal

    Parameters
    ----------
    H : array_like of shape (M,)
        (Complex) frequency response values
    N : int
        FIR filter order
    tau : int
        delay of filter
    f : array_like of shape (M,)
        frequencies at which ``H`` is given
    Fs : float
        sampling frequency of digital FIR filter
    Wt : array_like of shape (M,) or shape (M,M), optional
        vector of weights for a weighted least-squares method (default results in no
        weighting)

    Returns
    -------
    b : np.ndarray of shape (N+1,)
        The FIR filter coefficient vector in a 1-D sequence

    References
    ----------
    * Elster and Link [Elster2008]_

    .. see_also ::mod::`PyDynamic.uncertainty.propagate_filter.FIRuncFilter`

    """
    print(
        f"\ninvLSFIR: Least-squares fit of an order {N} digital FIR filter to the "
        f"reciprocal of a frequency response H given by {len(H)} values.\n"
    )

    frequencies = f.copy()
    sampling_frequency = Fs

    n_frequencies = len(frequencies)
    h_complex = _assemble_complex_from_real_imag(H)

    omega = _compute_radial_freqs_equals_two_pi_times_freqs_over_sampling_freq(
        sampling_frequency, frequencies
    )

    weights = _validate_and_return_weights(weights=Wt, expected_len=2 * n_frequencies)

    x = _compute_x(N, frequencies, sampling_frequency, weights)

    delayed_h_complex_recipr = np.reciprocal(
        h_complex * _compute_e_to_the_one_j_omega_tau(omega, tau)
    )
    iRI = _assemble_real_imag_from_complex(delayed_h_complex_recipr)

    return np.linalg.lstsq(x, iRI, rcond=None)[0]


def invLSFIR_unc(
    H: np.ndarray,
    UH: np.ndarray,
    N: int,
    tau: int,
    f: np.ndarray,
    Fs: float,
    wt: Optional[np.ndarray] = None,
    verbose: Optional[bool] = True,
    inv: Optional[bool] = True,
    trunc_svd_tol: Optional[float] = None,
):
    """Design of FIR filter as fit to freq. resp. or its reciprocal with uncertainties

    Least-squares fit of a (time-discrete) digital FIR filter to the reciprocal of a
    given frequency response for which associated uncertainties are given for its
    real and imaginary part. Uncertainties are propagated using a truncated svd
    and linear matrix propagation.

    Parameters
    ----------
    H : array_like of shape (M,) or (2M,)
        (Complex) frequency response values in dtype complex or as a vector first
        containing the real followed by the imaginary parts
    UH : array_like of shape (2M,2M)
        uncertainties associated with the real and imaginary part of H
    N : int
        FIR filter order
    tau : int
        time delay of filter in samples
    f : array_like of shape (M,)
        frequencies at which H is given
    Fs : float
        sampling frequency of digital FIR filter
    wt : array_like of shape (2M,), optional
        vector of weights for a weighted least-squares method (default results in no
        weighting)
    verbose : bool, optional
        whether to print statements to the command line (default = True)
    inv : bool, optional
        If False (default) apply the fit to the frequency response values directly,
        otherwise fit to the reciprocal of the frequency response values
    trunc_svd_tol : float, optional
        lower bound for singular values to be considered for pseudo-inverse

    Returns
    -------
    b : array_like of shape (N+1,)
        The FIR filter coefficient vector in a 1-D sequence
    Ub : array_like of shape (N+1,N+1)
        uncertainties associated with b

    References
    ----------
    * Elster and Link [Elster2008]_

    .. see_also ::mod::`PyDynamic.uncertainty.propagate_filter.FIRuncFilter`
    """
    if not inv:
        raise NotImplementedError(
            f"\ninvLSFIR_unc: The least-squares fitting of an order {N} digital FIR "
            f"filter to a frequency response H given by {len(H)} values with "
            f"propagation of associated uncertainties is not yet implemented. "
            f"Let us know, if this feature is of interest to you."
        )

    if verbose:
        print(
            f"\ninvLSFIR_unc: Least-squares fit of an order {N} digital FIR filter "
            f"to the reciprocal of a frequency response H given by {len(H)} values "
            f"and propagation of associated uncertainties."
        )

    frequencies = f.copy()
    sampling_frequency = Fs
    n_frequencies = len(frequencies)

    # Step 1: Propagation of uncertainties to reciprocal of frequency response
    runs = 10000

    if not len(H) == UH.shape[0]:
        # Assume that H is given as complex valued frequency response.
        RI = _assemble_real_imag_from_complex(H)
        h_complex = H.copy()
    else:
        RI = H.copy()
        h_complex = _assemble_complex_from_real_imag(H)

    h_complex_recipr = np.reciprocal(h_complex)
    HRI = np.random.multivariate_normal(RI, UH, runs)  # random draws of real,imag of

    omega = _compute_radial_freqs_equals_two_pi_times_freqs_over_sampling_freq(
        sampling_frequency, frequencies
    )
    omega_tau = omega * tau

    # Vectorized Monte Carlo for propagation to inverse
    absHMC = HRI[:, :n_frequencies] ** 2 + HRI[:, n_frequencies:] ** 2
    HiMC = np.hstack(
        (
            (
                HRI[:, :n_frequencies] * np.tile(np.cos(omega_tau), (runs, 1))
                + HRI[:, n_frequencies:] * np.tile(np.sin(omega_tau), (runs, 1))
            )
            / absHMC,
            (
                HRI[:, n_frequencies:] * np.tile(np.cos(omega_tau), (runs, 1))
                - HRI[:, :n_frequencies] * np.tile(np.sin(omega_tau), (runs, 1))
            )
            / absHMC,
        )
    )
    UiH = np.cov(HiMC, rowvar=False)

    # Step 2: Fit filter coefficients and evaluate uncertainties
    wt = _validate_and_return_weights(weights=wt, expected_len=2 * n_frequencies)

    X = _compute_x(N, frequencies, sampling_frequency, wt)
    Hri = _assemble_real_imag_from_complex(
        np.reciprocal(h_complex * _compute_e_to_the_one_j_omega_tau(omega, tau))
    )

    u, s, v = np.linalg.svd(X, full_matrices=False)
    if isinstance(trunc_svd_tol, float):
        s[s < trunc_svd_tol] = 0.0
    StSInv = np.zeros_like(s)
    StSInv[s > 0] = s[s > 0] ** (-2)

    M = np.dot(np.dot(np.dot(v.T, np.diag(StSInv)), np.diag(s)), u.T)

    filter_coeffs = np.dot(M, Hri[:, np.newaxis]).flatten()
    filter_coeffs_uncertainties = np.dot(np.dot(M, UiH), M.T)

    if verbose:
        Hd = _compute_delayed_filters_freq_resp_via_scipys_freqz(
            filter_coeffs, 1.0, tau, omega
        )
        residuals_real_imag = _assemble_real_imag_from_complex(
            Hd - np.reciprocal(h_complex_recipr)
        )
        _compute_and_print_rms(residuals_real_imag)

    return filter_coeffs, filter_coeffs_uncertainties


def _compute_and_print_rms(residuals_real_imag: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.sum(residuals_real_imag ** 2) / (len(residuals_real_imag) // 2))
    print(
        f"{_get_first_public_caller()}: Calculation of filter coefficients finished. "
        f"Final rms error = {rms}"
    )
    return rms


def invLSFIR_uncMC(
    H: np.ndarray,
    N: int,
    f: np.ndarray,
    Fs: float,
    tau: int,
    weights: Optional[np.ndarray] = None,
    verbose: Optional[bool] = True,
    inv: Optional[bool] = False,
    UH: Optional[np.ndarray] = None,
    mc_runs: Optional[int] = None,
    trunc_svd_tol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design of FIR filter as fit to freq. resp. or its reciprocal with uncertainties

    Least-squares fit of a (time-discrete) digital FIR filter to the reciprocal of the
    frequency response values or actual frequency response values for which
    associated uncertainties are given for its real and imaginary part. Uncertainties
    are propagated either using a Monte Carlo method if mc_runs is provided as
    integer greater than one or otherwise using a truncated singular-value
    decomposition and linear matrix propagation. The Monte Carlo approach may help in
    cases where the weighting matrix or the Jacobian are ill-conditioned, resulting
    in false uncertainties associated with the filter coefficients. Additionally it
    is needed to propagate uncertainties when fitting to the frequency response
    directly.

    Parameters
    ----------
    H : array_like of shape (M,) or (2M,)
        (Complex) frequency response values in dtype complex or as a vector
        containing the real parts in the first half followed by the imaginary parts
    N : int
        FIR filter order
    f : array_like of shape (M,)
        frequencies at which H is given
    Fs : float
        sampling frequency of digital FIR filter
    tau : int
        time delay in samples for improved fitting
    weights : array_like of shape (2M,), optional
        vector of weights for a weighted least-squares method (default results in no
        weighting)
    verbose: bool, optional
        If True (default) verbose output is printed to the command line
    inv : bool, optional
        If False (default) apply the fit to the frequency response values directly,
        otherwise fit to the reciprocal of the frequency response values
    UH : array_like of shape (2M,2M), optional
        uncertainties associated with the real and imaginary part of H
    mc_runs : int, optional
        Number of Monte Carlo runs greater than one. Only used, if uncertainties
        associated with the real and imaginary part of H are provided. Only one of
        mc_runs and trunc_svd_tol can be provided.
    trunc_svd_tol : float, optional
        Lower bound for singular values to be considered for pseudo-inverse. Values
        smaller than this threshold are considered zero. Defaults to zero. Only one of
        mc_runs and trunc_svd_tol can be provided.

    Returns
    -------
    b : array_like of shape (N+1,)
        The FIR filter coefficient vector in a 1-D sequence
    Ub : array_like of shape (N+1,N+1)
        Uncertainties associated with b.  Will be None if UH is not
        provided or is None.

    References
    ----------
    * Elster and Link [Elster2008]_

    Raises
    ------
    NotImplementedError
        The least-squares fitting of a digital FIR filter to a frequency response H
        with propagation of associated uncertainties using a truncated singular-value
        decomposition and linear matrix propagation is not yet implemented.
        Alternatively specify the number mc_runs of runs to propagate the uncertainties
        via the Monte Carlo method.
    """
    (
        freq_resps_real_imag,
        freqs,
        mc_runs,
        propagation_method,
        sampling_freq,
        weights,
    ) = _validate_and_prepare_fir_inputs(
        Fs, H, UH, f, inv, mc_runs, trunc_svd_tol, weights
    )
    if verbose:
        _print_fir_welcome_msg(H, N, inv, mc_runs, propagation_method, trunc_svd_tol)
    omega, delayed_freq_resp_real_imag_or_recipr, x = _prepare_common_fitting_inputs(
        N, freq_resps_real_imag, freqs, inv, sampling_freq, tau, weights
    )
    if propagation_method == _PropagationMethod.NONE:
        b_fir = _fit_filter_coeffs_via_least_squares(
            x, delayed_freq_resp_real_imag_or_recipr
        )
        Ub_fir = None
    else:
        mc_freq_resps_real_imag = _draw_multivariate_monte_carlo_samples(
            vector=freq_resps_real_imag, covariance_matrix=UH, mc_runs=mc_runs
        )
        if propagation_method == _PropagationMethod.MC:
            b_fir, Ub_fir = _fit_fir_filter_with_uncertainty_propagation_via_mc(
                inv, mc_freq_resps_real_imag, omega, tau, x
            )
        else:
            b_fir, Ub_fir = _fit_fir_filter_with_uncertainty_propagation_via_svd(
                mc_freq_resps_real_imag,
                mc_runs,
                omega,
                delayed_freq_resp_real_imag_or_recipr,
                tau,
                trunc_svd_tol,
                x,
            )
    if verbose:
        _print_fir_result_msg(b_fir, freq_resps_real_imag, inv, omega, tau)
    return b_fir, Ub_fir


class _PropagationMethod(Enum):
    NONE = 0
    MC = 1
    SVD = 2


def _validate_and_prepare_fir_inputs(
    sampling_freq: float,
    H: np.ndarray,
    UH: np.ndarray,
    freqs: np.ndarray,
    inv: bool,
    mc_runs: int,
    trunc_svd_tol: float,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, _PropagationMethod, float, np.ndarray]:
    n_freqs = len(freqs)
    two_n_freqs = 2 * n_freqs
    sampling_freq = sampling_freq
    weights = _validate_and_return_weights(weights, expected_len=two_n_freqs)
    freq_resps_real_imag = _validate_and_assemble_freq_resps(
        H,
        expected_len_when_complex=n_freqs,
        expected_len_when_real_imag=two_n_freqs,
    )
    _validate_uncertainties(vector=freq_resps_real_imag, covariance_matrix=UH)
    _validate_fir_uncertainty_propagation_method_related_inputs(
        UH, inv, mc_runs, trunc_svd_tol
    )
    propagation_method, mc_runs = _determine_fir_propagation_method(UH, mc_runs)
    return (
        freq_resps_real_imag,
        freqs,
        mc_runs,
        propagation_method,
        sampling_freq,
        weights,
    )


def _validate_and_return_weights(weights: np.ndarray, expected_len: int) -> np.ndarray:
    if weights is not None:
        if not isinstance(weights, np.ndarray):
            raise TypeError(
                f"{_get_first_public_caller()}: User-defined weighting has wrong "
                "type. wt is expected to be a NumPy ndarray but is of type "
                f"{type(weights)}.",
            )
        if len(weights) != expected_len:
            raise ValueError(
                f"{_get_first_public_caller()}: User-defined weighting has wrong "
                "dimension. wt is expected to be of length "
                f"{expected_len} but is of length {len(weights)}."
            )
        return weights
    return np.ones(expected_len)


def _get_first_public_caller() -> str:
    for caller in inspect.stack():
        if "PyDynamic" in caller.filename and caller.function[0] != "_":
            return caller.function
    return inspect.stack()[1].function


def _validate_and_assemble_freq_resps(
    H: np.ndarray,
    expected_len_when_complex: int,
    expected_len_when_real_imag: int,
) -> np.ndarray:
    if _is_dtype_complex(H):
        _validate_length_of_h(H, expected_length=expected_len_when_complex)
        return _assemble_real_imag_from_complex(H)
    _validate_length_of_h(H, expected_length=expected_len_when_real_imag)
    return H


def _is_dtype_complex(array: np.ndarray) -> bool:
    return array.dtype == np.complexfloating


def _validate_length_of_h(H: np.ndarray, expected_length: int):
    if not len(H) == expected_length:
        raise ValueError(
            f"{_get_first_public_caller()}: vector of complex frequency responses "
            f"is expected to contain {expected_length} elements, corresponding to the "
            f"number of frequencies, but actually contains {len(H)} "
            f"elements. Please adjust either of the two inputs."
        )


def _validate_uncertainties(vector: np.ndarray, covariance_matrix: np.ndarray):
    if not _no_uncertainties_were_provided(covariance_matrix):
        _validate_vector_and_corresponding_uncertainties_dims(
            vector=vector, covariance_matrix=covariance_matrix
        )


def _no_uncertainties_were_provided(covariance_matrix: Union[np.ndarray, None]) -> bool:
    return covariance_matrix is None


def _validate_vector_and_corresponding_uncertainties_dims(
    vector: np.ndarray, covariance_matrix: Union[np.ndarray]
):
    if not isinstance(covariance_matrix, np.ndarray):
        raise TypeError(
            f"{_get_first_public_caller()}: if uncertainties are provided, "
            f"they are expected to be of type np.ndarray, but uncertainties are of type"
            f" {type(covariance_matrix)}."
        )
    if not number_of_rows_equals_vector_dim(matrix=covariance_matrix, vector=vector):
        raise ValueError(
            f"{_get_first_public_caller()}: number of rows of uncertainties and "
            f"number of elements of values are expected to match. But {len(vector)} "
            f"values and {covariance_matrix.shape} uncertainties were provided. Please "
            f"adjust either the values or their corresponding uncertainties."
        )
    if not is_2d_square_matrix(covariance_matrix):
        raise ValueError(
            f"{_get_first_public_caller()}: uncertainties are expected to be "
            f"provided in a square matrix shape but are of shape "
            f"{covariance_matrix.shape}."
        )


def _validate_fir_uncertainty_propagation_method_related_inputs(
    covariance_matrix: Union[np.ndarray, None],
    inv: bool,
    mc_runs: Union[int, None],
    trunc_svd_tol: Union[float, None],
):
    def _are_we_supposed_to_apply_method_on_freq_resp_directly() -> bool:
        return not inv

    def _both_propagation_methods_simultaneously_requested() -> bool:
        return _input_for_svd_was_provided(
            trunc_svd_tol
        ) and _number_of_monte_carlo_runs_was_provided(mc_runs)

    def _are_we_supposed_to_fit_freq_resp_with_svd_propagation() -> bool:
        return (
            _input_for_svd_was_provided(trunc_svd_tol)
            or not _number_of_monte_carlo_runs_was_provided(mc_runs)
        ) and _are_we_supposed_to_apply_method_on_freq_resp_directly()

    def _number_of_mc_runs_too_small():
        return mc_runs == 1

    if _no_uncertainties_were_provided(covariance_matrix):
        if _number_of_monte_carlo_runs_was_provided(mc_runs):
            raise ValueError(
                f"\n{_get_first_public_caller()}: The least-squares fitting of a "
                f"digital FIR filter "
                "to a frequency response H with propagation of associated "
                f"uncertainties via the Monte Carlo method requires that uncertainties "
                f"are provided via input parameter UH. No uncertainties were given "
                f"but number of Monte Carlo runs set to {mc_runs}. Either remove "
                f"mc_runs or provide uncertainties."
            )
        if _input_for_svd_was_provided(trunc_svd_tol):
            raise ValueError(
                f"\n{_get_first_public_caller()}: The least-squares fitting of a "
                f"digital FIR filter "
                "to a frequency response H with propagation of associated "
                "uncertainties via a truncated singular-value decomposition and linear "
                "matrix propagation requires that uncertainties are provided via "
                "input parameter UH. No uncertainties were given but lower bound for "
                f"singular values trunc_svd_tol={trunc_svd_tol}. Either remove "
                "trunc_svd_tol or provide uncertainties."
            )
    elif _both_propagation_methods_simultaneously_requested():
        raise ValueError(
            f"\n{_get_first_public_caller()}: Only one of mc_runs and trunc_svd_tol "
            f"can be "
            f"provided but mc_runs={mc_runs} and trunc_svd_tol={trunc_svd_tol}."
        )
    elif _are_we_supposed_to_fit_freq_resp_with_svd_propagation():
        raise NotImplementedError(
            f"\n{_get_first_public_caller()}: The least-squares fitting of a digital "
            f"FIR filter to a frequency response H with propagation of associated "
            f"uncertainties using a truncated singular-value decomposition and linear "
            f"matrix propagation is not yet implemented. Alternatively specify "
            f"the number mc_runs of runs to propagate the uncertainties via the "
            f"Monte Carlo method."
        )
    elif _number_of_mc_runs_too_small():
        raise ValueError(
            f"\n{_get_first_public_caller()}: Number of Monte Carlo runs is expected "
            f"to be greater than 1 but mc_runs={mc_runs}. Please provide a greater "
            f"number of runs or switch to propagation of uncertainties "
            f"via singular-value decomposition by leaving out mc_runs."
        )


def _number_of_monte_carlo_runs_was_provided(mc_runs: Union[int, None]) -> bool:
    return bool(mc_runs)


def _input_for_svd_was_provided(trunc_svd_tol: Union[float, None]) -> bool:
    return trunc_svd_tol is not None


def _determine_fir_propagation_method(
    covariance_matrix: Union[np.ndarray, None],
    mc_runs: Union[int, None],
) -> Tuple[_PropagationMethod, Union[int, None]]:
    if _no_uncertainties_were_provided(covariance_matrix):
        return _PropagationMethod.NONE, None
    if _number_of_monte_carlo_runs_was_provided(mc_runs):
        return _PropagationMethod.MC, mc_runs
    return _PropagationMethod.SVD, 10000


def _print_fir_welcome_msg(
    freq_resp_in_provided_shape,
    filter_order,
    inv,
    mc_runs,
    propagation_method,
    trunc_svd_tol,
):
    if propagation_method != _PropagationMethod.NONE:
        if propagation_method == _PropagationMethod.MC:
            method_specific_propagation_msg = f"Monte Carlo method with {mc_runs} runs"
        else:
            method_specific_propagation_msg = (
                f"truncated singular-value decomposition and linear matrix "
                f"propagation with {trunc_svd_tol} as lower bound for the singular "
                f"values to be considered for the pseudo-inverse"
            )
        propagation_msg = (
            ". The frequency response's associated uncertainties are propagated "
            "via a " + method_specific_propagation_msg
        )
    else:
        propagation_msg = " without propagation of associated uncertainties"
    print(
        f"\ninvLSFIR_uncMC: Least-squares fit of an order {filter_order} digital FIR "
        f"filter to{' the reciprocal of' if inv else ''} a frequency response given by "
        f"{len(freq_resp_in_provided_shape)} values{propagation_msg}."
    )


def _prepare_common_fitting_inputs(
    filter_order: int,
    freq_resps_real_imag: np.ndarray,
    freqs: np.ndarray,
    inv: bool,
    sampling_freq: float,
    tau: int,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _compute_x(filter_order, freqs, sampling_freq, weights)
    omega = _compute_radial_freqs_equals_two_pi_times_freqs_over_sampling_freq(
        sampling_freq, freqs
    )
    complex_freq_resp = _assemble_complex_from_real_imag(freq_resps_real_imag)
    delayed_complex_freq_resp = complex_freq_resp * _compute_e_to_the_one_j_omega_tau(
        omega, tau
    )
    delayed_freq_resp_real_imag_or_recipr = (
        _assemble_delayed_freq_resp_real_imag_or_recipr(delayed_complex_freq_resp, inv)
    )
    return omega, delayed_freq_resp_real_imag_or_recipr, x


def _compute_e_to_the_one_j_omega_tau(omega: np.ndarray, tau: int) -> np.ndarray:
    return np.exp(1j * omega * tau)


def _assemble_delayed_freq_resp_real_imag_or_recipr(
    delayed_complex_freq_resp: np.ndarray, inv: bool
) -> np.ndarray:
    return _assemble_real_imag_from_complex(
        np.reciprocal(delayed_complex_freq_resp) if inv else delayed_complex_freq_resp
    )


def _fit_filter_coeffs_via_least_squares(
    x: np.ndarray, delayed_freq_resp_real_imag_or_recipr: np.ndarray
) -> np.ndarray:
    return np.linalg.lstsq(x, delayed_freq_resp_real_imag_or_recipr, rcond=None)[0]


def _draw_multivariate_monte_carlo_samples(
    vector: np.ndarray, covariance_matrix: np.ndarray, mc_runs: int
) -> np.ndarray:
    return multivariate_normal.rvs(mean=vector, cov=covariance_matrix, size=mc_runs)


def _fit_fir_filter_with_uncertainty_propagation_via_mc(
    inv: bool,
    mc_freq_resps_real_imag: np.ndarray,
    omega: np.ndarray,
    tau: int,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mc_delayed_freq_resps_or_recipr_real_imag = (
        _compute_mc_delayed_freq_resps_or_reciprs_real_imag(
            inv, mc_freq_resps_real_imag, omega, tau
        )
    )
    return _conduct_fir_uncertainty_propagation_via_mc(
        mc_delayed_freq_resps_or_recipr_real_imag, x
    )


def _compute_mc_delayed_freq_resps_or_reciprs_real_imag(
    inv: bool, mc_freq_resps_real_imag: np.ndarray, omega: np.ndarray, tau: int
) -> np.ndarray:
    mc_complex_freq_resps = _assemble_complex_from_real_imag(mc_freq_resps_real_imag)
    mc_delayed_complex_freq_resps = (
        mc_complex_freq_resps * _compute_e_to_the_one_j_omega_tau(omega, tau)
    )
    return _assemble_delayed_freq_resp_real_imag_or_recipr(
        mc_delayed_complex_freq_resps, inv
    )


def _conduct_fir_uncertainty_propagation_via_mc(
    mc_freq_resps_real_imag: np.ndarray, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mc_b_firs = np.array(
        [
            _fit_filter_coeffs_via_least_squares(x, mc_freq_resp)
            for mc_freq_resp in mc_freq_resps_real_imag
        ]
    ).T
    b_fir = np.mean(mc_b_firs, axis=1)
    Ub_fir = np.cov(mc_b_firs, rowvar=True)
    return b_fir, Ub_fir


def _fit_fir_filter_with_uncertainty_propagation_via_svd(
    mc_freq_resps_real_imag: np.ndarray,
    mc_runs: int,
    omega: np.ndarray,
    preprocessed_freq_resp: np.ndarray,
    tau: int,
    trunc_svd_tol: float,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    list_of_mc_freq_resps_real_and_imag = (
        _split_array_of_monte_carlo_samples_of_real_and_imag_parts(
            mc_freq_resps_real_imag
        )
    )
    mc_freq_resps_real = list_of_mc_freq_resps_real_and_imag[0]
    mc_freq_resps_imag = list_of_mc_freq_resps_real_and_imag[1]
    recipr_of_sqr_abs_of_mc_freq_resps = np.reciprocal(
        mc_freq_resps_real ** 2 + mc_freq_resps_imag ** 2
    )
    omega_tau = omega * tau
    cos_omega_tau = np.tile(np.cos(omega_tau), (mc_runs, 1))
    sin_omega_tau = np.tile(np.sin(omega_tau), (mc_runs, 1))
    UiH = np.cov(
        np.hstack(
            (
                (
                    mc_freq_resps_real * cos_omega_tau
                    + mc_freq_resps_imag * sin_omega_tau
                )
                * recipr_of_sqr_abs_of_mc_freq_resps,
                (
                    mc_freq_resps_imag * cos_omega_tau
                    - mc_freq_resps_real * sin_omega_tau
                )
                * recipr_of_sqr_abs_of_mc_freq_resps,
            )
        ),
        rowvar=False,
    )
    u, s, v = np.linalg.svd(x, full_matrices=False)
    if _input_for_svd_was_provided(trunc_svd_tol):
        s[s < trunc_svd_tol] = 0.0
    StSInv = np.zeros_like(s)
    StSInv[s > 0] = s[s > 0] ** (-2)
    M = np.dot(np.dot(np.dot(v.T, np.diag(StSInv)), np.diag(s)), u.T)
    b_fir = np.dot(M, preprocessed_freq_resp[:, np.newaxis]).flatten()
    Ub_fir = np.dot(np.dot(M, UiH), M.T)
    return b_fir, Ub_fir


def _print_fir_result_msg(
    b_fir: np.ndarray,
    freq_resps_real_imag: np.ndarray,
    inv: bool,
    omega: np.ndarray,
    tau: int,
):
    complex_h = _assemble_complex_from_real_imag(freq_resps_real_imag)
    original_values = np.reciprocal(complex_h) if inv else complex_h
    delayed_filters_freq_resp = _compute_delayed_filters_freq_resp_via_scipys_freqz(
        b_fir, 1.0, tau, omega
    )
    complex_residuals = delayed_filters_freq_resp - original_values
    residuals_real_imag = _assemble_real_imag_from_complex(complex_residuals)
    _compute_and_print_rms(residuals_real_imag)


def invLSIIR(H, Nb, Na, f, Fs, tau, justFit=False, verbose=True):
    """Least-squares IIR filter fit to the reciprocal of given frequency response values

    Least-squares fit of a digital IIR filter to the reciprocal of a given set
    of frequency response values and stabilization by pole mapping and introduction
    of a time delay.

    Parameters
    ----------
    H : array_like of shape (M,)
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
        return LSIIR(H, Nb, Na, f, Fs, tau, verbose, 0, True)
    return LSIIR(H, Nb, Na, f, Fs, tau, verbose, 50, True)


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
    return LSIIR(H, Nb, Na, f, Fs, tau, False, 50, True, UH, 1000)
