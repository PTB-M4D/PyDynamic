'''
.. moduleauthor:: Sascha Eichstaedt (sascha.eichstaedt@ptb.de)
'''
from numpy.random import RandomState
from numpy import diff, sqrt, sum, array, corrcoef
import numpy as np
from scipy.misc import comb
from scipy.signal import periodogram


def shocklikeGaussian(time, t0, m0, sigma, noise = 0.0):
	"""
    Generates a signal that resembles a shock excitation as a Gaussian followed by a smaller Gaussian of opposite sign.
	Args:
	    time - numpy array of time instants (equidistant)
	    t0   - time instant of signal maximum
	    m0   - signal maximum
	    sigma- std of main pulse
	    noise- std of simulated signal noise (optional)

	Returns:
	    x - signal amplitudes at time instants

    """

	sigm1 = sigma  # width of first pulse
	sigm2 = 1.2 * sigma  # width of second pulse
	x = m0 * np.exp(-(time - t0) ** 2 / (2 * sigm1 ** 2)) - m0 * 0.5 * np.exp(
		-(time - t0 * 1.1) ** 2 / (2 * sigm2 ** 2))
	if noise > 0:
		x += np.random.randn(len(time)) * noise
	return x


def GaussianPulse(time, t0, m0, sigma, noise = 0.0):
	"""
    Generates a Gaussian pulse at t0 with height m0 and std sigma
    Args:
        time - numpy array of time instants (equidistant)
        t0   - time instant of signal maximum
        m0   - signal maximum
        sigma- std of pulse
        noise- std of simulated signal noise (optional)

    Returns:
        x - signal amplitudes at time instants
    """

	x = m0 * np.exp(-(time - t0) ** 2 / (2 * sigma ** 2))
	if noise > 0:
		x = x + np.random.randn(len(time)) * noise
	return x


def rect(time, t0, t1, height, noise = 0.0):
	"""
    Generates a rectangular signal of given height and width t1-t0

    Args:
        time   - numpy array of time instants (equidistant)
        t0     - time instant of rect lhs
        t1     - time instant of rect rhs
        height - signal maximum
        noise  - std of simulated signal noise (optional)
    Returns:
        x  - signal amplitudes at time instants
    """

	x = np.zeros((len(time),))
	x[np.nonzero(time > t0)] = height
	x[np.nonzero(time > t1)] = 0.0

	if noise > 0:
		x = x + np.random.randn(len(time)) * noise
	return x


def GaussianPulseDeriv(time, t0, m0, sigma):
	"""
	Pulse like signal as derivative of a Gaussian pulse

	Args:
	    time  - numpy array of time instants (equidistant)
	    t0    - time instant of signal maximum
		m0    - maximum of Gaussian pulse
		sigma - std of Gaussian pulse

	Returns:
	     x  - signal amplitudes at time instants
	"""

	return -2.0 * m0 * (time - t0) / (2 * sigma ** 2) * np.exp(-(time - t0) ** 2 / (2 * sigma ** 2))


def squarepulse(time, height, numpulse = 4, noise = 0.0):
	"""
    Generates a series of rect functions to represent a square pulse signal

    Args:
        time - numpy array of time instants
        height - height of the rectangular pulses
        numpulse - number of pulses
        noise - std of simulated signal noise (optional)
    Returns:
        x - signal amplitude at time instants
    """
	width = (time[-1] - time[0]) / (2 * numpulse + 1)  # width of each individual rect
	x = np.zeros_like(time)
	for k in range(numpulse):
		x += rect(time, (2 * k + 1) * width, (2 * k + 2) * width, height)
	if noise > 0:
		x += np.random.randn(len(time)) * noise
	return x


class corr_noise(object):
	def __init__(self, w, sigma, seed = None):
		self.w = w
		self.sigma = sigma
		self.rst = RandomState(seed)

	def calc_noise(self, N = 100):
		z = self.rst.randn(N + 4)
		noise = diff(diff(diff(diff(z * self.w ** 4) - 4 * z[1:] * self.w ** 3) + 6 * z[2:] * self.w ** 2) - 4 * z[
																												 3:] * self.w) + z[
																																 4:]
		self.Cw = sqrt(sum([comb(4, l) ** 2 * self.w ** (2 * l) for l in range(5)]))
		self.noise = noise * self.sigma / self.Cw
		return self.noise

	def calc_noise2(self, N = 100):
		P = np.ceil(1.5 * N)
		NT = self.rst.randn(P) * self.sigma
		STD = np.zeros(21)
		STD[10] = 1.0
		for counter in range(5):
			NTtmp = NT.copy()
			NT[:-1] = NT[:-1] + self.w * NTtmp[1:]
			NT[-1] = NT[-1] + self.w * NTtmp[-1]
			NT[1:] = NT[1:] + self.w * NTtmp[:-1]
			NT[0] = NT[0] + self.w * NTtmp[-1]
			STDtmp = STD.copy()
			STD[1:] = STD[1:] + self.w * STDtmp[:-1]
			STD[:-1] = STD[:-1] + self.w * STDtmp[1:]
		NT = NT / np.linalg.norm(STD)
		self.noise = NT[:N]
		self.Cw = sqrt(sum([comb(4, l) ** 2 * self.w ** (2 * l) for l in range(5)]))
		return self.noise

	def calc_autocorr(self, lag = 10):
		return array([1] + [corrcoef(self.noise[:-i], self.noise[i:])[0, 1] for i in range(1, lag)])

	def calc_cov(self):
		def cw(k):
			if np.abs(k) > 4: return 0
			c = sum([comb(4, l) * comb(4, np.abs(k) + l) * self.w ** (np.abs(k) + 2 * l) for l in range(5 - np.abs(k))])
			return c / self.Cw ** 2

		N = len(self.noise)
		Sigma = np.zeros((N, N))
		for m in range(N):
			for n in range(m, N):
				Sigma[m, n] = self.sigma ** 2 * cw(n - m)
		self.Sigma = Sigma + Sigma.T - np.diag(np.diag(Sigma))
		return self.Sigma

	def calc_psd(self, noise = None, Fs = 1.0, **kwargs):
		if isinstance(noise, np.ndarray):
			return periodogram(noise, fs = Fs, **kwargs)
		else:
			return periodogram(self.noise, fs = Fs, **kwargs)
