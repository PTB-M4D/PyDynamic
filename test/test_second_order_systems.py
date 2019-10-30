# -*- coding: utf-8 -*-
""" Perform tests on methods specialized for second order dynamic systems."""

from matplotlib.pyplot import np as np
from pytest import approx
from scipy.signal import freqs

from PyDynamic.misc.SecondOrderSystem import sos_FreqResp, sos_phys2filter, \
    sos_realimag, sos_absphase

np.random.seed(12345)

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
    assert np.round(np.abs(f0 - fe5[indmax])) <= 0.01 * f0
    assert np.abs(H[0]) == approx(S0)
    Hmulti = sos_FreqResp(
        S0 * np.ones(K), delta * np.ones(K), f0 * np.ones(K), fe5)
    assert Hmulti.shape[1] == K


def test_sos_phys2filter():
    b, a = sos_phys2filter(S0, delta, f0)
    H = freqs(b, a, 2 * np.pi * fe5)[1]
    indmax = np.abs(H).argmax()
    assert np.round(np.abs(f0 - fe5[indmax])) <= 0.01 * f0
    assert np.abs(H[0]) == approx(S0)
    bmulti, amulti = sos_phys2filter(
        S0 * np.ones(K), delta * np.ones(K), f0 * np.ones(K))
    assert len(bmulti[0]) == K
    assert amulti.shape == (K, 3)


def test_sos_realimag():
    Hmean, Hcov = sos_realimag(S0, delta, f0, uS0, udelta, uf0, fe3, runs=100)
    assert Hcov.shape, (2 * len(fe3) == 2 * len(fe3))
    H = sos_FreqResp(S0, delta, f0, fe3)
    assert np.linalg.norm(H) == approx(np.linalg.norm(Hmean))


def test_sos_absphase():
    Hmean, Hcov = sos_absphase(S0, delta, f0, uS0, udelta, uf0, fe3, runs=100)
    assert Hcov.shape == (2 * len(fe3), 2 * len(fe3))
    H = sos_FreqResp(S0, delta, f0, fe3)
    assert np.linalg.norm(H) == approx(np.linalg.norm(Hmean))
