import unittest
import numpy as np
print(__name__)


from PyDynamic.misc.SecondOrderSystem import sos_FreqResp
from PyDynamic.identification import fit_filter


class Test_LSIIR(unittest.TestCase):
	def test_LSIIR(self):
		# measurement system
		f0 = 36e3           # system resonance frequency in Hz
		S0 = 0.124          # system static gain
		delta = 0.0055      # system damping

		f = np.linspace(0, 80e3, 30)               # frequencies for fitting the system
		Hvals = sos_FreqResp(S0, delta, f0, f)      # frequency response of the 2nd order system

		#%% fitting the IIR filter

		Fs = 500e3          # sampling frequency
		Na = 4; Nb = 4     # IIR filter order (Na - denominator, Nb - numerator)

		b, a, tau = fit_filter.LSIIR(Hvals, Na, Nb, f, Fs)

		self.assertEqual(len(b), Nb+1)
		self.assertEqual(len(a), Na+1)
		self.assertTrue(isinstance(tau,int))



if __name__ == '__main__':
	unittest.main()
