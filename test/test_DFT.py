# -*- coding: utf-8 -*-
""" Perform tests on methods to handle DFT and inverse DFT."""

import numpy as np
from numpy.testing import assert_almost_equal
from pytest import approx

from PyDynamic.misc.testsignals import multi_sine
from PyDynamic.uncertainty.propagate_DFT import *


def multisine_testsignal():
    # set amplitude values of multi-sine componentens (last parameter is number of components)
    sine_amps = np.random.randint(1, 4, 10)
    # set frequencies of multi-sine components
    sine_freqs = np.linspace(100, 500, len(sine_amps)) * 2 * np.pi
    # define time axis
    dt = 0.0001
    time = np.arange(0.0, 0.2, dt)
    # measurement noise standard deviation (assume white noise)
    sigma_noise = 0.001
    # generate test signal
    testsignal = multi_sine(time, sine_amps, sine_freqs, noise=sigma_noise)
    return testsignal, sigma_noise


class TestDFTmethods:
    def test_AmpPhasePropagation(self):
        testsignal, noise_std = multisine_testsignal()
        A, P, UAP = Time2AmpPhase(testsignal, noise_std ** 2)
        x, ux = AmpPhase2Time(A, P, UAP)
        assert_almost_equal(np.max(np.abs(testsignal-x)),0)



