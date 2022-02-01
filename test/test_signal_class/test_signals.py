import numpy as np
import pytest
from hypothesis import given, strategies as hst
from numpy.testing import assert_almost_equal
from pytest import approx

from PyDynamic.misc.testsignals import (
    GaussianPulse,
    multi_sine,
    rect,
    shocklikeGaussian,
    sine,
    squarepulse,
)
from ..conftest import (
    hypothesis_not_negative_float,
)


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


@given(hst.floats(), hst.floats(), hypothesis_not_negative_float(allow_infinity=True))
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
