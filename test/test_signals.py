# -*- coding: utf-8 -*-
""" Perform tests on methods to create testsignals."""
import matplotlib
import numpy as np
import pytest
from hypothesis import given, strategies as hst
from numpy.testing import assert_almost_equal
from pytest import approx

from examples.working_with_signals import demonstrate_signal
from PyDynamic.misc.testsignals import (
    GaussianPulse,
    multi_sine,
    rect,
    shocklikeGaussian,
    sine,
    squarepulse,
)


@pytest.fixture(scope="module")
def N():
    return 2048


@pytest.fixture(scope="module")
def Ts():
    return 0.01


@pytest.fixture(scope="module")
def hi_res_time(create_timestamps):
    return create_timestamps(0, 2 * np.pi, 1e-5)


@pytest.fixture(scope="module")
def create_timestamps(N, Ts):
    def timestamps(t_0=0, t_n=N * Ts, d_t=Ts):
        # Compute equally spaced timestamps between t_0 and t_n.
        return np.arange(t_0, t_n, d_t)

    return timestamps


@pytest.fixture(scope="module")
def time(create_timestamps, N, Ts):
    return create_timestamps(0, N * Ts, Ts)


@pytest.fixture(scope="module")
def t0(N, Ts):
    return N / 2 * Ts


def test_shocklikeGaussian(Ts, t0, N, time):
    m0 = 1 + np.random.rand() * 0.2
    sigma = 50 * Ts
    # zero noise
    x = shocklikeGaussian(time, t0, m0, sigma, noise=0.0)
    assert_almost_equal(np.max(x), m0)
    assert np.std(x[: N // 10]) < 1e-10
    # noisy signal
    nstd = 1e-2
    x = shocklikeGaussian(time, t0, m0, sigma, noise=nstd)
    assert_almost_equal(np.round(np.std(x[: N // 10]) * 100) / 100, nstd)


def test_GaussianPulse(Ts, t0, N, time):
    m0 = 1 + np.random.rand() * 0.2
    sigma = 50 * Ts
    # zero noise
    x = GaussianPulse(time, t0, m0, sigma, noise=0.0)
    assert_almost_equal(np.max(x), m0)
    assert_almost_equal(time[x.argmax()], t0)
    assert np.std(x[: N // 10]) < 1e-10
    # noisy signal
    nstd = 1e-2
    x = GaussianPulse(time, t0, m0, sigma, noise=nstd)
    assert_almost_equal(np.round(np.std(x[: N // 10]) * 100) / 100, nstd)


def test_rect(Ts, t0, N, time):
    width = N // 4 * Ts
    height = 1 + np.random.rand() * 0.2
    x = rect(time, t0, t0 + width, height, noise=0.0)
    assert_almost_equal(np.max(x), height)
    assert np.max(x[time < t0]) < 1e-10
    assert np.max(x[time > t0 + width]) < 1e-10
    # noisy signal
    nstd = 1e-2
    x = rect(time, t0, t0 + width, height, noise=nstd)
    assert np.round(np.std(x[time < t0]) * 100) / 100 == approx(nstd)


def test_squarepulse(time):
    height = 1 + np.random.rand() * 0.2
    numpulses = 5
    x = squarepulse(time, height, numpulses, noise=0.0)
    assert_almost_equal(np.max(x), height)


@pytest.mark.slow
def test_minimal_call_max_sine(time):
    x = sine(time)
    # Check for minimal callability and that maximum amplitude at
    # timestamps is below default.
    assert np.max(np.abs(x)) <= 1.0


@pytest.mark.slow
def test_minimal_call_hi_res_max_sine(hi_res_time):
    x = sine(hi_res_time)
    # Check for minimal callability with high resolution time vector and
    # that maximum amplitude at timestamps is almost equal default.
    assert_almost_equal(np.max(x), 1.0)
    assert_almost_equal(np.min(x), -1.0)


@given(
    hst.floats(min_value=1, max_value=1e64, allow_infinity=False, allow_nan=False),
    hst.integers(min_value=1, max_value=1000),
)
@pytest.mark.slow
def test_medium_call_freq_multiples_sine(
    time, hi_res_time, create_timestamps, freq, rep
):
    # Create time vector with timestamps near multiples of frequency.
    fixed_freq_time = create_timestamps(time[0], rep * 1 / freq, 1 / freq)
    x = sine(fixed_freq_time, freq=freq)
    # Check if signal at multiples of frequency is start value of signal.
    for i_x in x:
        assert_almost_equal(i_x, 0)


@given(hst.floats(min_value=0, exclude_min=True, allow_infinity=False))
@pytest.mark.slow
def test_medium_call_max_sine(time, amp):
    # Test if casual time signal's maximum equals the input amplitude.

    x = sine(time, amp=amp)
    # Check for minimal callability and that maximum amplitude at
    # timestamps is below default.
    assert np.max(np.abs(x)) <= amp


@given(hst.floats(min_value=0, exclude_min=True, allow_infinity=False))
@pytest.mark.slow
def test_medium_call_hi_res_max_sine(hi_res_time, amp):
    # Test if high-resoluted time signal's maximum equals the input amplitude.

    # Initialize fixed amplitude.
    x = sine(hi_res_time, amp=amp)
    # Check for minimal callability with high resolution time vector and
    # that maximum amplitude at timestamps is almost equal default.
    assert_almost_equal(np.max(x), amp)
    assert_almost_equal(np.min(x), -amp)


@given(hst.floats(), hst.floats(), hst.floats())
def test_full_call_sine(time, amp, freq, noise):
    # Check for all possible calls.
    sine(time)
    sine(time, amp)
    sine(time, amp=amp)
    sine(time, amp, freq)
    sine(time, amp, freq=freq)
    sine(time, amp=amp, freq=freq)
    sine(time, freq=freq)
    sine(time, amp, freq, noise=noise)
    sine(time, noise=noise)
    sine(time, amp, freq=freq, noise=noise)
    sine(time, freq=freq, noise=noise)
    sine(time, amp=amp, freq=freq, noise=noise)


@given(hst.floats(), hst.floats())
def test_compare_multisine_with_sine(time, freq, amp):
    # Compare the result of a call of sine and a similar call of multi_sine
    # with one-element lists of amplitudes and frequencies.

    x = sine(time=time, amp=amp, freq=freq)
    multi_x = multi_sine(time=time, amps=[amp], freqs=[freq])
    # Check for minimal callability and that maximum amplitude at
    # timestamps is below default.
    assert_almost_equal(x, multi_x)


@pytest.mark.slow
def test_signal_example(monkeypatch):
    # Test executability of the demonstrate_signal example.
    # With this expression we override the matplotlib.pyplot.show method with a
    # lambda expression returning None but only for this one test.
    monkeypatch.setattr(matplotlib.pyplot, "show", lambda: None, raising=True)
    demonstrate_signal()
