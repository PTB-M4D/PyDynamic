# -*- coding: utf-8 -*-
"""
Perform tests on methods specialized for second order dynamic systems.
"""

from matplotlib.pyplot import np as np
from nose.tools import assert_equal, assert_almost_equal, assert_less_equal
from scipy.signal import freqs

# import numpy as np
from PyDynamic.misc.SecondOrderSystem import sos_FreqResp, sos_phys2filter, sos_realimag, sos_absphase

Fs = 100e3
delta = 0.0001
f0 = float(Fs / 4 + np.abs(np.random.randn(1)) * Fs / 8)
S0 = 1.0
fe5 = np.linspace(0, Fs / 2, 100000)
K = 100
fe3 = np.linspace(0, Fs / 2, 1000)
udelta = 1e-12 * delta
uf0 = 1e-12 * f0
uS0 = 1e-12 * S0


def test_sos_freqresp():
    H = sos_FreqResp(S0, delta, f0, fe5)
    indmax = np.abs(H).argmax()
    assert_less_equal(np.round(np.abs(f0 - fe5[indmax])), 0.01 * f0)
    assert_almost_equal(np.abs(H[0]), S0, places=8)
    Hmulti = sos_FreqResp(S0 * np.ones(K), delta * np.ones(K), f0 * np.ones(K), fe5)
    assert_equal(Hmulti.shape[1], K)


def test_sos_phys2filter():
    b, a = sos_phys2filter(S0, delta, f0)
    H = freqs(b, a, 2 * np.pi * fe5)[1]
    indmax = np.abs(H).argmax()
    assert_less_equal(np.round(np.abs(f0 - fe5[indmax])), 0.01 * f0)
    assert_almost_equal(np.abs(H[0]), S0, places=8)
    bmulti, amulti = sos_phys2filter(S0 * np.ones(K), delta * np.ones(K), f0 * np.ones(K))
    assert_equal(len(bmulti[0]), K)
    assert_equal(amulti.shape, (K, 3))


def test_sos_realimag():
    Hmean, Hcov = sos_realimag(S0, delta, f0, uS0, udelta, uf0, fe3, runs=100)
    assert_equal(Hcov.shape, (2 * len(fe3), 2 * len(fe3)))
    H = sos_FreqResp(S0, delta, f0, fe3)
    assert_almost_equal(np.linalg.norm(H), np.linalg.norm(Hmean), places=5)


def test_sos_absphase():
    Hmean, Hcov = sos_absphase(S0, delta, f0, uS0, udelta, uf0, fe3, runs=100)
    assert_equal(Hcov.shape, (2 * len(fe3), 2 * len(fe3)))
    H = sos_FreqResp(S0, delta, f0, fe3)
    assert_almost_equal(np.linalg.norm(H), np.linalg.norm(Hmean), places=5)
