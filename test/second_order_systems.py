# -*- coding: utf-8 -*-

import unittest
from PyDynamic.misc.SecondOrderSystem import sos_FreqResp, sos_phys2filter, sos_realimag, sos_absphase
from scipy.signal import freqs

from matplotlib.pyplot import *

class sos(unittest.TestCase):
	def test_sos_freqresp(self):
		Fs = 100e3
		delta = 0.0001
		f0 = float(Fs/4 + np.abs(np.random.randn(1))*Fs/8)
		S0 = 1.0
		f = np.linspace(0, Fs/2, 100000)
		H = sos_FreqResp(S0, delta, f0, f)
		indmax = np.abs(H).argmax()
		self.assert_(np.round(np.abs(f0-f[indmax]))<=0.01*f0)
		self.assertAlmostEqual(np.abs(H[0]), S0, places = 8)

		K = 100
		Hmulti = sos_FreqResp(S0*np.ones(K), delta*np.ones(K), f0*np.ones(K), f)
		self.assertEquals(Hmulti.shape[1], K)

	def test_sos_phys2filter(self):
		Fs = 100e3
		delta = 0.0001
		f0 = float(Fs / 4 + np.abs(np.random.randn(1)) * Fs / 8)
		S0 = 1.0
		f = np.linspace(0, Fs / 2, 100000)
		b,a = sos_phys2filter(S0, delta, f0)
		H = freqs(b, a, 2*np.pi*f)[1]
		indmax = np.abs(H).argmax()
		self.assert_(np.round(np.abs(f0-f[indmax]))<=0.01*f0)
		self.assertAlmostEqual(np.abs(H[0]), S0, places = 8)

		K = 100
		bmulti, amulti = sos_phys2filter(S0 * np.ones(K), delta * np.ones(K), f0 * np.ones(K))
		self.assertEquals(len(bmulti[0]), K)
		self.assertEquals(amulti.shape, (K,3))

	def test_sos_realimag(self):
		Fs = 100e3
		delta = 0.0001
		f0 = float(Fs / 4 + np.abs(np.random.randn(1)) * Fs / 8)
		S0 = 1.0
		udelta = 1e-12*delta
		uf0 = 1e-12*f0
		uS0 = 1e-12*S0
		f = np.linspace(0, Fs / 2, 1000)
		Hmean, Hcov = sos_realimag(S0, delta, f0, uS0, udelta, uf0, f, runs = 100)
		self.assertEqual(Hcov.shape, (2*len(f), 2*len(f)))
		H = sos_FreqResp(S0, delta, f0, f)
		self.assertAlmostEqual(np.linalg.norm(H), np.linalg.norm(Hmean), places = 5)


	def test_sos_absphase(self):
		Fs = 100e3
		delta = 0.0001
		f0 = float(Fs / 4 + np.abs(np.random.randn(1)) * Fs / 8)
		S0 = 1.0
		udelta = 1e-12 * delta
		uf0 = 1e-12 * f0
		uS0 = 1e-12 * S0
		f = np.linspace(0, Fs / 2, 1000)
		Hmean, Hcov = sos_absphase(S0, delta, f0, uS0, udelta, uf0, f, runs = 100)
		self.assertEqual(Hcov.shape, (2 * len(f), 2 * len(f)))
		H = sos_FreqResp(S0, delta, f0, f)
		self.assertAlmostEqual(np.linalg.norm(H), np.linalg.norm(Hmean), places = 5)


if __name__ == '__main__':
	unittest.main()
