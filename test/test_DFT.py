# -*- coding: utf-8 -*-
""" Perform tests on methods to handle DFT and inverse DFT."""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import pytest
from hypothesis import assume, given, strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite
from numpy.testing import assert_allclose, assert_almost_equal

from PyDynamic.misc.testsignals import multi_sine
from PyDynamic.uncertainty.propagate_DFT import (
    _apply_window,
    AmpPhase2Time,
    GUM_DFT,
    GUM_iDFT,
    Time2AmpPhase,
)


@pytest.fixture
def multisine_testsignal() -> Tuple[np.ndarray, float]:
    """Additional helper function to create test multi-sine signal"""
    dt = 0.0001
    # set amplitude values of multi-sine components (last parameter is number of
    # components)
    sine_amps = np.random.randint(1, 4, 10)
    # set frequencies of multi-sine components
    sine_freqs = np.linspace(100, 500, len(sine_amps)) * 2 * np.pi
    # define time axis
    time = np.arange(0.0, 0.2, dt)
    time = time[: int(2 ** np.floor(np.log2(len(time))))]
    # measurement noise standard deviation (assume white noise)
    sigma_noise = 0.001
    # generate test signal
    testsignal = multi_sine(time, sine_amps, sine_freqs, noise=sigma_noise)
    return testsignal, sigma_noise


@pytest.fixture(scope="module")
def create_corrmatrix():
    def _create_corrmatrix(rho, Nx, nu=0.5, phi=0.3):
        """ Additional helper function to create a correlation matrix"""
        corrmat = np.zeros((Nx, Nx))
        if rho > 1:
            raise ValueError("Correlation scalar should be less than one.")

        for k in range(1, Nx):
            corrmat += np.diag(np.ones(Nx - k) * rho ** (phi * k ** nu), k)
        corrmat += corrmat.T
        corrmat += np.eye(Nx)

        return corrmat
    return _create_corrmatrix


class TestDFT:
    def test_DFT_iDFT(self, multisine_testsignal):
        """Test GUM_DFT and GUM_iDFT with noise variance as uncertainty"""
        x, ux = multisine_testsignal
        X, UX = GUM_DFT(x, ux ** 2)
        xh, uxh = GUM_iDFT(X, UX)
        assert_almost_equal(np.max(np.abs(x - xh)), 0)
        assert_almost_equal(np.max(ux - np.sqrt(np.diag(uxh))), 0)

    def test_DFT_iDFT_vector(self, multisine_testsignal):
        """Test GUM_DFT and GUM_iDFT with uncertainty vector"""
        x, ux = multisine_testsignal
        ux = (0.1 * x) ** 2
        X, UX = GUM_DFT(x, ux)
        xh, uxh = GUM_iDFT(X, UX)
        assert_almost_equal(np.max(np.abs(x - xh)), 0)
        assert_almost_equal(np.max(np.sqrt(ux) - np.sqrt(np.diag(uxh))), 0)

    def test_DFT_iDFT_fullcov(self, multisine_testsignal, create_corrmatrix):
        """Test GUM_DFT and GUM_iDFT with full covariance matrix"""
        x, ux = multisine_testsignal
        ux = np.ones_like(x) * 0.01 ** 2
        cx = create_corrmatrix(0.95, len(x))
        Ux = np.diag(ux)
        Ux = Ux.dot(cx.dot(Ux))
        X, UX = GUM_DFT(x, Ux)
        xh, Uxh = GUM_iDFT(X, UX)
        assert_almost_equal(np.max(np.abs(x - xh)), 0)
        assert_almost_equal(np.max(Ux - Uxh), 0)

    def test_AmpPhasePropagation(self, multisine_testsignal):
        """Test Time2AmpPhase and AmpPhase2Time with noise variance as uncertainty"""
        testsignal, noise_std = multisine_testsignal
        A, P, UAP = Time2AmpPhase(testsignal, noise_std ** 2)
        x, ux = AmpPhase2Time(A, P, UAP)
        assert_almost_equal(np.max(np.abs(testsignal - x)), 0)


@pytest.fixture
def known_inputs_and_outputs_for_apply_window():
    x = np.array(
        [
            1.81912799e-051,
            -np.inf,
            1.81912799e-051,
            -7.86008596e245,
            1.81912799e-051,
            1.81912799e-051,
            0.00000000e000,
            -9.00719925e015,
            1.81912799e-051,
            1.81912799e-051,
        ]
    )
    Ux = 1.192092896e-07
    window = np.array(
        [
            -1.50000000e000,
            -3.42473686e-168,
            -3.42473686e-168,
            -3.42473686e-168,
            0.00000000e000,
            1.19209290e-007,
            -3.42473686e-168,
            -3.42473686e-168,
            -3.42473686e-168,
            -3.42473686e-168,
        ]
    )
    xw = np.array(
        [
            -2.72869198e-051,
            np.inf,
            -6.23003466e-219,
            2.69187261e078,
            0.00000000e000,
            2.16856955e-058,
            -0.00000000e000,
            3.08472873e-152,
            -6.23003466e-219,
            -6.23003466e-219,
        ]
    )
    Uxw = np.array([2.68220902e-07, 0, 0, 0, 0, 1.69406590e-21, 0, 0, 0, 0])
    return {"inputs": {"x": x, "Ux": Ux, "window": window}, "outputs": (xw, Uxw)}


@composite
def x_Ux_and_window(draw: Callable, ux_type: Optional[Union[np.ndarray, float]] = None):
    """Provide sample x, associated uncertainties and a random window

    Parameters
    ----------
    draw : internal Hypothesis drawing caller (see
        https://hypothesis.readthedocs.io/en/latest/data.html?highlight=composite
        for details)
    ux_type : either np.ndarray or str, optional
        Specification for desired uncertainties as desired return type

    Returns
    -------
    Dictionary to hand over to apply_window(**x_Ux_and_window([...])).
    """
    # Draw dimension of x, window and of Ux in case it is a matrix.
    dim = draw(hst.integers(min_value=1, max_value=20))

    x = draw(hnp.arrays(dtype=float, shape=dim))
    # Prepare drawing Ux as matrix if requested or either as float or matrix else.
    full_ux_strategy = hnp.arrays(dtype=float, shape=(dim, dim))
    float_ux_strategy = hst.floats(min_value=0)
    if ux_type == np.ndarray:
        uncertainty_strategy = full_ux_strategy
    elif ux_type == float:
        uncertainty_strategy = float_ux_strategy
    else:
        uncertainty_strategy = hst.one_of(float_ux_strategy, full_ux_strategy)
    Ux = draw(uncertainty_strategy)
    window = draw(hnp.arrays(dtype=float, shape=dim))

    return {"x": x, "Ux": Ux, "window": window}


@given(x_Ux_and_window())
def test_apply_window(params):
    """Check if application of window to random sample signal works"""
    assert _apply_window(**params)


@given(x_Ux_and_window())
def test_wrong_dimension_x_apply_window(params):
    """Check that wrong dimension of x causes exception"""
    # Make sure x contains more than one value and then break it.
    assume(len(params["x"]) > 1)
    params["x"] = params["x"][1:]

    with pytest.raises(AssertionError):
        _apply_window(**params)


@given(x_Ux_and_window())
def test_wrong_dimension_window_apply_window(params):
    """Check that wrong dimension of x causes exception"""
    # Make sure x contains more than one value and then break it.
    assume(len(params["window"]) > 1)
    params["window"] = params["window"][1:]

    with pytest.raises(AssertionError):
        _apply_window(**params)


@given(x_Ux_and_window(ux_type=np.ndarray))
def test_wrong_dimension_ux_apply_window(params):
    """Check that wrong dimension of x causes exception"""
    # Make sure Ux is at least 2x2 and then break it.
    assume(params["Ux"].shape[0] > 1)

    # Cut Ux by one row. Now ux is not square anymore.
    params["Ux"] = params["Ux"][1:]

    with pytest.raises(AssertionError):
        _apply_window(**params)

    # Cut Ux by one column. Now ux is square again but does not match x.
    params["Ux"] = params["Ux"][:, 1:]

    with pytest.raises(AssertionError):
        _apply_window(**params)


@given(x_Ux_and_window(ux_type=float))
def test_apply_window(params):
    """Check if application of window to random sample with scalar uncertainty works"""
    assert _apply_window(**params)


def test_apply_window_with_known_result(known_inputs_and_outputs_for_apply_window):
    """Test one actually known result for apply_window"""
    assert_allclose(
        actual=_apply_window(**known_inputs_and_outputs_for_apply_window["inputs"]),
        desired=known_inputs_and_outputs_for_apply_window["outputs"],
    )
