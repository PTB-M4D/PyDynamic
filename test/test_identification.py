# -*- coding: utf-8 -*-
"""Perform tests on identification part of package model_estimation."""

import numpy as np
from hypothesis import example, given, strategies as hst
from numpy.testing import assert_almost_equal
import scipy.signal as dsp

from PyDynamic import grpdelay, mapinside, sos_FreqResp
from PyDynamic.model_estimation import fit_filter


@hst.composite
def LSIIR_parameters(draw):
    """Design a sample measurement system and a corresponding frequency response."""
    # Set the maximum absolute value for floats to be really unique in calculations.
    float_generic_params = {
        "allow_nan": False,
        "allow_infinity": False,
    }
    # measurement system
    f0 = draw(
        hst.floats(
            min_value=20e3,
            max_value=1e5,
            **float_generic_params,
        )
    )  # originally this was set to 36e3 for the system resonance frequency in Hz

    S0 = draw(
        hst.floats(
            min_value=0,
            max_value=1,
            **float_generic_params,
        )
    )  # originally this was set to 0.124 for the system static gain

    delta = draw(
        hst.floats(
            min_value=1e-4,
            max_value=1e-2,
            **float_generic_params,
        )
    )  # originally this was set to 0.0055 for the system damping
    dim = draw(
        hst.integers(
            min_value=1,
            max_value=30,
        )
    )  # originally this was set to 30 for the number of frequencies
    maximum_frequency = draw(
        hst.floats(
            min_value=1e3,
            max_value=1e5,
            **float_generic_params,
        )
    )  # originally this was set to 80e3 for the system damping
    f = np.linspace(0, maximum_frequency, dim)  # frequencies for fitting the system
    Hvals = sos_FreqResp(S0, delta, f0, f)  # frequency response of the 2nd order system

    # %% fitting the IIR filter

    Fs = draw(
        hst.floats(
            min_value=1e5,
            max_value=5e6,
            **float_generic_params,
        )
    )  # originally this was set to 500e3 for the sampling frequency
    Na = draw(
        hst.integers(
            min_value=1,
            max_value=10,
        )
    )  # originally this was set to 4 for the IIR denominator filter order
    Nb = draw(
        hst.integers(
            min_value=1,
            max_value=10,
        )
    )  # originally this was set to 4 for the IIR numerator filter order
    return {
        "Hvals": Hvals,
        "Na": Na,
        "Nb": Nb,
        "f": f,
        "Fs": Fs,
    }


def _former_fitIIR(
    _Hvals: np.ndarray,
    _tau: int,
    _w: np.ndarray,
    _E: np.ndarray,
    _Na: int,
    _Nb: int,
    _inv: bool = False,
):
    """The actual fitting routing for the least-squares IIR filter.

    Parameters
    ----------
        _Hvals :  (M,) np.ndarray
            (complex) frequency response values
        _tau : integer
            initial estimate of time delay
        _w : np.ndarray
            :math:`2 * np.pi * f / Fs`
        _E : np.ndarray
            :math:`np.exp(-1j * np.dot(w[:, np.newaxis], Ns.T))`
        _Nb : int
            numerator polynomial order
        _Na : int
            denominator polynomial order
        _inv : bool, optional
            If True the least-squares fitting is performed for the reciprocal,
            if False
            for the actual frequency response

    Returns
    -------
        b, a : IIR filter coefficients as numpy arrays
    """
    exponent = -1 if _inv else 1
    Ea = _E[:, 1 : _Na + 1]
    Eb = _E[:, : _Nb + 1]
    Htau = np.exp(-1j * _w * _tau) * _Hvals ** exponent
    HEa = np.dot(np.diag(Htau), Ea)
    D = np.hstack((HEa, -Eb))
    Tmp1 = np.real(np.dot(np.conj(D.T), D))
    Tmp2 = np.real(np.dot(np.conj(D.T), -Htau))
    ab = np.linalg.lstsq(Tmp1, Tmp2, rcond=None)[0]
    a_coeff = np.hstack((1.0, ab[:_Na]))
    b_coeff = ab[_Na:]
    return b_coeff, a_coeff


def _former_LSIIR(Hvals, Nb, Na, f, Fs, tau=0, justFit=False):
    """LSIIR method before version 2.0.0

    This helps to assure that the rewritten version matches the results of the
    previous implementation. We only commented out all statements, that did not
    contribute to the actual computation. This is the state in which the
    implementation was in Commit d2ac33ef4d5425de5bd1989d24fe9c11908f2aa0.

    Parameters
    ----------
        Hvals:   numpy array of (complex) frequency response values of shape (M,)
        Nb:      integer numerator polynomial order
        Na:      integer denominator polynomial order
        f:       numpy array of frequencies at which Hvals is given of shape
        (M,)
        Fs:      sampling frequency
        tau:     integer initial estimate of time delay
        justFit: boolean, when true then no stabilization is carried out

    Returns
    -------
        b,a:    IIR filter coefficients as numpy arrays
        tau:    filter time delay in samples

    References
    ----------
    * Eichstädt et al. 2010 [Eichst2010]_
    * Vuerinckx et al. 1996 [Vuer1996]_

    """

    def _fitIIR(
        _Hvals: np.ndarray,
        _tau: int,
        _w: np.ndarray,
        _E: np.ndarray,
        _Na: int,
        _Nb: int,
        _inv: bool = False,
    ):
        """The actual fitting routing for the least-squares IIR filter.

        Parameters
        ----------
            _Hvals :  (M,) np.ndarray
                (complex) frequency response values
            _tau : integer
                initial estimate of time delay
            _w : np.ndarray
                :math:`2 * np.pi * f / Fs`
            _E : np.ndarray
                :math:`np.exp(-1j * np.dot(w[:, np.newaxis], Ns.T))`
            _Nb : int
                numerator polynomial order
            _Na : int
                denominator polynomial order
            _inv : bool, optional
                If True the least-squares fitting is performed for the reciprocal,
                if False
                for the actual frequency response

        Returns
        -------
            b, a : IIR filter coefficients as numpy arrays
        """
        exponent = -1 if _inv else 1
        Ea = _E[:, 1 : _Na + 1]
        Eb = _E[:, : _Nb + 1]
        Htau = np.exp(-1j * _w * tau) * _Hvals ** exponent
        HEa = np.dot(np.diag(Htau), Ea)
        D = np.hstack((HEa, -Eb))
        Tmp1 = np.real(np.dot(np.conj(D.T), D))
        Tmp2 = np.real(np.dot(np.conj(D.T), -Htau))
        ab = np.linalg.lstsq(Tmp1, Tmp2, rcond=None)[0]
        a_coeff = np.hstack((1.0, ab[:_Na]))
        b_coeff = ab[_Na:]
        return b_coeff, a_coeff

    # print("\nLeast-squares fit of an order %d digital IIR filter" % max(Nb, Na))
    # print("to a frequency response given by %d values.\n" % len(Hvals))

    w = 2 * np.pi * f / Fs
    Ns = np.arange(0, max(Nb, Na) + 1)[:, np.newaxis]
    E = np.exp(-1j * np.dot(w[:, np.newaxis], Ns.T))

    b, a = _former_fitIIR(Hvals, tau, w, E, Na, Nb, _inv=False)

    if justFit:
        # print("Calculation done. No stabilization requested.")
        # if np.count_nonzero(np.abs(np.roots(a)) > 1) > 0:
        # print("Obtained filter is NOT stable.")
        # sos = np.sum(
        #    np.abs((dsp.freqz(b, a, 2 * np.pi * f / Fs)[1] - Hvals) ** 2))
        # print("Final sum of squares = %e" % sos)
        tau = 0
        return b, a, tau

    if np.count_nonzero(np.abs(np.roots(a)) > 1) > 0:
        stable = False
    else:
        stable = True

    maxiter = 50

    astab = mapinside(a)
    run = 1

    while stable is not True and run < maxiter:
        g1 = grpdelay(b, a, Fs)[0]
        g2 = grpdelay(b, astab, Fs)[0]
        tau = np.ceil(tau + np.median(g2 - g1))

        b, a = _former_fitIIR(Hvals, tau, w, E, Na, Nb, _inv=False)
        if np.count_nonzero(np.abs(np.roots(a)) > 1) > 0:
            astab = mapinside(a)
        else:
            stable = True
        run = run + 1

    # if np.count_nonzero(np.abs(np.roots(a)) > 1) > 0:
    # print("Caution: The algorithm did NOT result in a stable IIR filter!")
    # print(
    #    "Maybe try again with a higher value of tau0 or a higher filter "
    #    "order?")

    # print(
    #    "Least squares fit finished after %d iterations (tau=%d)." % (run, tau))
    # Hd = dsp.freqz(b, a, 2 * np.pi * f / Fs)[1]
    # Hd = Hd * np.exp(1j * 2 * np.pi * f / Fs * tau)
    # res = np.hstack((np.real(Hd) - np.real(Hvals), np.imag(Hd) - np.imag(Hvals)))
    # rms = np.sqrt(np.sum(res ** 2) / len(f))
    # print("Final rms error = %e \n\n" % rms)

    return b, a, int(tau)


@given(LSIIR_parameters())
def test_LSIIR_outputs_format(parameters):
    """This checks against expected formats of the outputs."""
    b, a, tau = fit_filter.LSIIR(**parameters)

    assert len(b) == parameters["Nb"] + 1
    assert len(a) == parameters["Na"] + 1
    assert isinstance(tau, int)
    assert tau >= 0


@given(LSIIR_parameters(), hst.integers(min_value=0, max_value=100), hst.booleans())
def test_fitIIR_results_against_former_implementations(LSIIR_parameters, tau, inv):
    """This takes the implementation prior to the rewrite and compares results."""
    # Initialize parameters.
    w = 2 * np.pi * LSIIR_parameters["f"] / LSIIR_parameters["Fs"]
    Ns = np.arange(0, max(LSIIR_parameters["Nb"], LSIIR_parameters["Na"]) + 1)[:, np.newaxis]
    E = np.exp(-1j * np.dot(w[:, np.newaxis], Ns.T))
    fit_params = {
        "Hvals": LSIIR_parameters["Hvals"],
        "tau": tau,
        "w": w,
        "E": E,
        "Na": LSIIR_parameters["Na"],
        "Nb": LSIIR_parameters["Nb"],
        "inv": inv,
    }

    # Compute solution of current version.
    b_current, a_current = fit_filter._fitIIR(**fit_params)

    # Rename parameter dict keys to the same name with leading underscore for the old
    # version.
    fit_params = {f"_{key}": value for key, value in fit_params.items()}

    # Compute solution of former version.
    b_former, a_former = _former_fitIIR(**fit_params)

    assert_almost_equal(b_current, b_former)
    assert_almost_equal(a_current, a_former)



@given(LSIIR_parameters())
def test_LSIIR_results_against_former_implementations(parameters):
    """This takes the implementation prior to the rewrite and compares results."""
    b_current, a_current, tau_current = fit_filter.LSIIR(**parameters)
    b_former, a_former, tau_former = _former_LSIIR(**parameters)

    assert_almost_equal(b_current, b_former)
    assert_almost_equal(a_current, a_former)
    assert_almost_equal(tau_current, tau_former)
