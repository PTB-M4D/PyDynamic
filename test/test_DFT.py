# -*- coding: utf-8 -*-
""" Perform tests on methods to handle DFT and inverse DFT."""

import numpy as np
from numpy.testing import assert_almost_equal
from pytest import approx

from PyDynamic.misc.testsignals import multi_sine
from PyDynamic.uncertainty.propagate_DFT import *

def prevpow2(n):
    return int(2 ** np.floor(np.log2(n)))

def multisine_testsignal(dt = 0.0001):
    """ Additional helper function to create test multi-sine signal
    """
    # set amplitude values of multi-sine componentens (last parameter is number of components)
    sine_amps = np.random.randint(1, 4, 10)
    # set frequencies of multi-sine components
    sine_freqs = np.linspace(100, 500, len(sine_amps)) * 2 * np.pi
    # define time axis
    time = np.arange(0.0, 0.2, dt)
    time = time[:prevpow2(len(time))]
    # measurement noise standard deviation (assume white noise)
    sigma_noise = 0.001
    # generate test signal
    testsignal = multi_sine(time, sine_amps, sine_freqs, noise=sigma_noise)
    return testsignal, sigma_noise

def create_corrmatrix(rho, Nx, nu=0.5, phi=0.3):
    """ Additional helper function to create a correlation matrix
    """
    corrmat = np.zeros((Nx,Nx))
    if rho > 1:
        raise ValueError("Correlation scalar should be less than one.")

    for k in range(1,Nx):
        corrmat += np.diag(np.ones(Nx-k)*rho**(phi*k**nu), k)
    corrmat += corrmat.T
    corrmat += np.eye(Nx)

    return corrmat

class TestDFTmethods:
    def test_DFT_iDFT(self):
        # test GUM_DFT and GUM_iDFT by calling it back and forth with noise variance as uncertainty
        x, ux = multisine_testsignal()
        X, UX = GUM_DFT(x, ux**2)
        xh, uxh = GUM_iDFT(X, UX)
        assert_almost_equal(np.max(np.abs(x-xh)),0)
        assert_almost_equal(np.max(ux - np.sqrt(np.diag(uxh))),0)

    def test_DFT_iDFT_vector(self):
        # test GUM_DFT and GUM_iDFT by calling it back and forth with uncertainty vector
        x, ux = multisine_testsignal()
        ux = (0.1 * x) ** 2
        X, UX = GUM_DFT(x, ux)
        xh, uxh = GUM_iDFT(X, UX)
        assert_almost_equal(np.max(np.abs(x-xh)),0)
        assert_almost_equal(np.max(np.sqrt(ux) - np.sqrt(np.diag(uxh))),0)

    def test_DFT_iDFT_fullcov(self):
        # test GUM_DFT and GUM_iDFT by calling it back and forth with full covariance matrix
        x, ux = multisine_testsignal()
        ux = np.ones_like(x)*0.01**2
        cx = create_corrmatrix(0.95, len(x))
        Ux = np.diag(ux)
        Ux = Ux.dot(cx.dot(Ux))
        X, UX = GUM_DFT(x, Ux)
        xh, Uxh = GUM_iDFT(X, UX)
        assert_almost_equal(np.max(np.abs(x-xh)),0)
        assert_almost_equal(np.max(Ux - Uxh),0)

    def test_AmpPhasePropagation(self):
        testsignal, noise_std = multisine_testsignal()
        A, P, UAP = Time2AmpPhase(testsignal, noise_std ** 2)
        x, ux = AmpPhase2Time(A, P, UAP)
        assert_almost_equal(np.max(np.abs(testsignal-x)),0)
