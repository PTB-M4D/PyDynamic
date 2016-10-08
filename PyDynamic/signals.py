# -*- coding: utf-8 -*-
"""
This module implements the signals class and its derivatives
Signals are dynamic quantities with associated uncertainties. A signal has to be defined together
with a time axis

Note: Abbreviation GUM stands for "Guide to the Expression of Uncertainty in Measurement"
"""

import numpy as np
from matplotlib.pyplot import *
import scipy.signal as dsp
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter, IIRuncFilter
from PyDynamic.uncertainty.propagate_MonteCarlo import SMC
from PyDynamic.misc.testsignals import rect

__all__ = ["Signal"]

class Signal():
	"""
	This is the base class which defines the interfaces and default behaviour
	"""
	unit_time = ""
	unit_values = ""
	name = ""
	def __init__(self, time, values, Ts=None, Fs=None, uncertainty=None):
		if len(values.shape)>1:
			raise NotImplementedError("Multivariate signals are not implemented yet.")
		assert(len(time)==len(values))
		self.time = time
		self.values = values
		# set sampling interval and frequency
		if (Ts is None) and (Fs is None):
			self.Ts = np.unique(np.diff(self.time)).mean()
			self.Fs = 1 / self.Ts
		elif isinstance(Ts, float):
			self.Ts = Ts
			if Fs is None:
				self.Fs = 1/Ts
			else:
				assert (np.abs(Fs*self.Ts-1)<1e-5),"Sampling interval and sampling frequency are inconsistent."
		# set initial uncertainty
		if isinstance(uncertainty, float):
			self.uncertainty = np.ones_like(values)*uncertainty
		elif isinstance(uncertainty, np.ndarray):
			uncertainty = uncertainty.squeeze()
			if len(uncertainty.shape)==1:
				assert(len(uncertainty)==len(time))
			else:
				assert(uncertainty.shape[0]==uncertainty.shape[1])
			self.uncertainty = uncertainty
		else:
			self.uncertainty = np.zeros_like(values)
		self.set_labels()

	def set_labels(self, unit_time=None, unit_values=None, name_values = None):
		if isinstance(unit_time,str):
			self.unit_time=unit_time
		else:
			self.unit_time = "s"
		if isinstance(unit_values,str):
			self.unit_values = unit_values
		else:
			self.unit_values = "a.u."
		if isinstance(name_values,str):
			self.name = name_values
		else:
			self.name = "signal"

	def plot(self, fignr=1, figsize=(10,8)):
		figure(fignr,figsize = figsize)
		plot(self.time, self.values, label=self.name)
		xlabel("time / %s"%self.unit_time)
		ylabel("%s / %s"%(self.name, self.unit_values))
		legend(loc="best")
		show()

	def apply_filter(self, b, a=1, Uab=None, MonteCarloRuns=None):
		"""Apply digital filter (b,a) to the signal values and propagate the uncertainty associated with the signal
		Parameters
		----------
			b: np.ndarray
				filter numerator coefficients
			a: np.ndarray
				filter denominator coefficients, use a=1 for FIR type filter
			Uab: np.ndarray
				covariance matrix associated with filter coefficients, Uab=None if no uncertainty associated with filter
			MonteCarloRuns: int
				number of Monte Carlo runs, if `None` then GUM linearization will be used
		Returns
		-------
			no return variables
		"""
		if not (isinstance(a, np.ndarray) or isinstance(a, list)): # FIR type filter
			if len(self.uncertainty.shape)==1:
				if not isinstance(MonteCarloRuns,int):
					self.values, self.uncertainty = \
						FIRuncFilter(self.values, self.uncertainty, b, Utheta = Uab)
				else:
					self.values, self.uncertainty = \
						SMC(self.values, self.uncertainty, b, a, Uab, runs=MonteCarloRuns)



if __name__=="__main__":
	N = 1024
	Ts = 0.01
	time = np.arange(0, N*Ts, Ts)
	x = rect(time, Ts*N//4, Ts*N//4*3)
	signal = Signal(time, x, Ts = Ts)
	b  = dsp.firls(15, [0, 0.2*signal.Fs/2, 0.25*signal.Fs/2, signal.Fs/2 ], [1, 1, 0, 0], nyq=signal.Fs/2)
	signal.apply_filter(b)
	signal.plot()