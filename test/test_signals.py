import numpy as np
import unittest
from PyDynamic.misc.testsignals import *

class Test_testsignals(unittest.TestCase):
	N = 2048
	Ts= 0.01
	time = np.arange(0, N*Ts, Ts)
	t0 = N/2*Ts

	def test_shocklikeGaussian(self):
		m0 = 1 + np.random.rand()*0.2
		sigma = 50*self.Ts
		# zero noise
		x = shocklikeGaussian(self.time, self.t0, m0, sigma, noise = 0.0)
		self.assertAlmostEqual(x.max(), m0, places = 5)
		self.assertLess(np.std(x[:self.N//10]), 1e-10)
		# noisy signal
		nstd = 1e-2
		x = shocklikeGaussian(self.time, self.t0, m0, sigma, noise = nstd)
		self.assertAlmostEqual(np.round(np.std(x[:self.N//10])*100)/100, nstd)

	def test_GaussianPulse(self):
		m0 = 1 + np.random.rand()*0.2
		sigma = 50*self.Ts
		# zero noise
		x = GaussianPulse(self.time, self.t0, m0, sigma, noise = 0.0)
		self.assertAlmostEqual(x.max(), m0)
		self.assertAlmostEqual(self.time[x.argmax()], self.t0)
		self.assertLess(np.std(x[:self.N//10]), 1e-10)
		# noisy signal
		nstd = 1e-2
		x = GaussianPulse(self.time, self.t0, m0, sigma, noise = nstd)
		self.assertAlmostEqual(np.round(np.std(x[:self.N//10])*100)/100, nstd)

	def test_rect(self):
		width = self.N//4*self.Ts
		height = 1 + np.random.rand()*0.2
		x = rect(self.time, self.t0, self.t0+width, height, noise = 0.0)
		self.assertAlmostEqual(x.max(), height)
		self.assertLess(np.max(x[self.time<self.t0]), 1e-10)
		self.assertLess(np.max(x[self.time>self.t0+width]), 1e-10)
		# noisy signal
		nstd = 1e-2
		x = rect(self.time, self.t0, self.t0+width, height, noise = nstd)
		self.assertAlmostEqual(np.round(np.std(x[self.time<self.t0]) * 100) / 100, nstd)

	def test_squarepulse(self):
		height = 1 + np.random.rand()*0.2
		numpulses = 5
		x = squarepulse(self.time, height, numpulses, noise = 0.0)
		self.assertAlmostEqual(x.max(), height)


if __name__ == '__main__':
	unittest.main()
