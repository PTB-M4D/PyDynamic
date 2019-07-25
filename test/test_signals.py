# -*- coding: utf-8 -*-
""" Perform tests on methods to create testsignals."""

import numpy as np
from numpy.testing import assert_almost_equal
from pytest import approx

from PyDynamic.misc.testsignals import shocklikeGaussian, GaussianPulse, rect, \
    squarepulse, sine

N = 2048
Ts = 0.01


def get_timestamps(t_0=0, t_n=N * Ts, d_t=Ts):
    # Compute equally spaced timestamps between t_0 and t_n.
    return np.arange(t_0, t_n, d_t)


time = get_timestamps(0, N * Ts, Ts)
t0 = N / 2 * Ts


def test_shocklikeGaussian():
    m0 = 1 + np.random.rand() * 0.2
    sigma = 50 * Ts
    # zero noise
    x = shocklikeGaussian(time, t0, m0, sigma, noise=0.0)
    assert_almost_equal(np.max(x), m0)
    assert np.std(x[:N // 10]) < 1e-10
    # noisy signal
    nstd = 1e-2
    x = shocklikeGaussian(time, t0, m0, sigma, noise=nstd)
    assert_almost_equal(
        np.round(np.std(x[:N // 10]) * 100) / 100, nstd)


def test_GaussianPulse():
    m0 = 1 + np.random.rand() * 0.2
    sigma = 50 * Ts
    # zero noise
    x = GaussianPulse(time, t0, m0, sigma, noise=0.0)
    assert_almost_equal(np.max(x), m0)
    assert_almost_equal(time[x.argmax()], t0)
    assert np.std(x[:N // 10]) < 1e-10
    # noisy signal
    nstd = 1e-2
    x = GaussianPulse(time, t0, m0, sigma, noise=nstd)
    assert_almost_equal(
        np.round(np.std(x[:N // 10]) * 100) / 100, nstd)


def test_rect():
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


def test_squarepulse():
    height = 1 + np.random.rand() * 0.2
    numpulses = 5
    x = squarepulse(time, height, numpulses, noise=0.0)
    assert_almost_equal(np.max(x), height)


class TestSine:
    # Test the sine signal.
    hi_res_time = get_timestamps(0, 2 * np.pi, 1e-5)

    def test_minimal_call_max_sine(self):
        x = sine(time)
        # Check for minimal callability and that maximum amplitude at
        # timestamps is below default.
        assert np.max(np.abs(x)) <= 1.0

    def test_minimal_call_hi_res_max_sine(self):
        x = sine(self.hi_res_time)
        # Check for minimal callability with high resolution time vector and
        # that maximum amplitude at timestamps is almost equal default.
        assert_almost_equal(np.max(x), 1.0)
        assert_almost_equal(np.min(x), -1.0)

    def test_medium_call_freq_multiples_sine(self):
        # Initialize fixed frequency and number of repetitions.
        np.random.seed(130)
        freq = 100 * np.random.rand()
        rep = 10 * freq
        # Create time vector with timestamps near multiples of frequency.
        fixed_freq_time = get_timestamps(time[0], rep, freq)
        x = sine(fixed_freq_time, freq=freq)
        # Check if signal at multiples of frequency is start value of signal.
        for i_x in x:
            assert_almost_equal(i_x, 0)

    def test_medium_call_max_sine(self):
        # Initialize fixed amplitude.
        np.random.seed(11201)
        amp = np.random.rand()
        x = sine(time, amp=amp)
        # Check for minimal callability and that maximum amplitude at
        # timestamps is below default.
        assert np.max(np.abs(x)) <= amp

    def test_medium_call_hi_res_max_sine(self):
        # Initialize fixed amplitude.
        np.random.seed(1101)
        amp = np.random.rand()
        x = sine(self.hi_res_time, amp=amp)
        # Check for minimal callability with high resolution time vector and
        # that maximum amplitude at timestamps is almost equal default.
        assert_almost_equal(np.max(x), amp)
        assert_almost_equal(np.min(x), -amp)

    def test_full_call_sine(self):
        # Check for all possible calls.
        np.random.seed(1121)
        amp = np.random.rand()
        freq = 100 * np.random.rand()
        noise = np.random.rand()
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
