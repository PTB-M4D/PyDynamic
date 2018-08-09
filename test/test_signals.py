# -*- coding: utf-8 -*-
"""
Perform tests on methods to create testsignals.
"""

import numpy as np
from nose.tools import assert_almost_equal, assert_less

from PyDynamic.misc.testsignals import shocklikeGaussian, GaussianPulse, rect, squarepulse

N = 2048
Ts = 0.01
time = np.arange(0, N * Ts, Ts)
t0 = N / 2 * Ts


def test_shocklikeGaussian():
    m0 = 1 + np.random.rand() * 0.2
    sigma = 50 * Ts
    # zero noise
    x = shocklikeGaussian(time, t0, m0, sigma, noise=0.0)
    assert_almost_equal(x.max(), m0, places=5)
    assert_less(np.std(x[:N // 10]), 1e-10)
    # noisy signal
    nstd = 1e-2
    x = shocklikeGaussian(time, t0, m0, sigma, noise=nstd)
    assert_almost_equal(np.round(np.std(x[:N // 10]) * 100) / 100, nstd)


def test_GaussianPulse():
    m0 = 1 + np.random.rand() * 0.2
    sigma = 50 * Ts
    # zero noise
    x = GaussianPulse(time, t0, m0, sigma, noise=0.0)
    assert_almost_equal(x.max(), m0)
    assert_almost_equal(time[x.argmax()], t0)
    assert_less(np.std(x[:N // 10]), 1e-10)
    # noisy signal
    nstd = 1e-2
    x = GaussianPulse(time, t0, m0, sigma, noise=nstd)
    assert_almost_equal(np.round(np.std(x[:N // 10]) * 100) / 100, nstd)


def test_rect():
    width = N // 4 * Ts
    height = 1 + np.random.rand() * 0.2
    x = rect(time, t0, t0 + width, height, noise=0.0)
    assert_almost_equal(x.max(), height)
    assert_less(np.max(x[time < t0]), 1e-10)
    assert_less(np.max(x[time > t0 + width]), 1e-10)
    # noisy signal
    nstd = 1e-2
    x = rect(time, t0, t0 + width, height, noise=nstd)
    assert_almost_equal(np.round(np.std(x[time < t0]) * 100) / 100, nstd)


def test_squarepulse():
    height = 1 + np.random.rand() * 0.2
    numpulses = 5
    x = squarepulse(time, height, numpulses, noise=0.0)
    assert_almost_equal(x.max(), height)
