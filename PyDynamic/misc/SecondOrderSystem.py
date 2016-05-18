# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Sascha Eichstaedt (sascha.eichstaedt@ptb.de)

A collection of modules and methods that are used throughout the whole package.
Methods specialized for second order dynamic systems, such as the ones used
for high-class accelerometers.

"""

import numpy as np
if __name__=="misc.SecondOrderSystem": # module is imported from within package
	from misc.filterstuff import ua
	from misc.tools import col_hstack
else:
	from ..misc.filterstuff import ua
	from ..misc.tools import col_hstack


def FreqResp(S, d, f0, freqs):
	""" Calculation of the system frequency response

	The frequency response is calculated from the continuous physical model
	of a second order system such as

	:math:`H(f) = \\frac{4S\\pi^2f_0^2}{(2\\pi f_0)^2 + 2jd(2\\pi f_0)f - f^2}`

	If the provided system parameters are vectors then :math:`H(f)` is calculated for
	each set of parameters.

	Parameters:
		S:      float or ndarray shape (K,)
				static gain
		d:      float or ndarray shape (K,)
				damping parameter
		f0:     float or ndarray shape (K,)
				resonance frequency
		freqs:  ndarray shape (N,)
				frequencies at which to calculate the freq response

	Returns:
		H:  ndarray shape (N,) or ndarray shape (K,N)
			complex frequency response values

	"""
	om0 = 2 * np.pi * f0
	rho = S * (om0 ** 2)
	w = 2 * np.pi * freqs


	if isinstance(S,np.ndarray):
		H = np.tile(rho,(len(w),1))*( om0**2 + 2j*np.tile(d*om0,(len(w),1))*np.tile(w[:,np.newaxis],(1,len(S))) - np.tile(w[:,np.newaxis]**2,(1,len(S))) )**(-1)
	else:
		H = rho/(om0**2 + 2j*d*om0*w - w**2)

	return H


def phys2filter(S, d, f0):
	"""Calculation of continuous filter coefficients from physical parameters.

	If the provided system parameters are vectors then the filter coefficients
	are calculated for each set of parameters.

	Parameters:
		S:  float
			static gain
		d:  float
			damping parameter
		f0: float
			resonance frequency

	Returns:
		b,a: ndarray
			 analogue filter coefficients

	"""

	if isinstance(S,np.ndarray):
		bc = S*(2*np.pi*f0)**2
		ac = col_hstack([np.ones((len(S),)),4*d*np.pi*f0,(2*np.pi*f0)**2])
	else:
		bc = S*(2*np.pi*f0)**2
		ac = np.array([1, 2 * d * 2 * np.pi * f0, (2 * np.pi * f0) ** 2])

	return bc,ac


def unc_realimag(S, d, f0, uS, ud, uf0, f):
	"""Propagation of uncertainty from physical parameters to real and imaginary
	part of system's transfer function using GUM S2 Monte Carlo.

	Parameters:
		S:    float
			  static gain
		d:    float
			  damping
		f0:   float
			  resonance frequency
		uS:   float
			  uncertainty associated with static gain
		ud:   float
			  uncertainty associated with damping
		uf0:  float
			  uncertainty associated with resonance frequency
		f:    ndarray, shape (N,)
			  frequency values at which to calculate real and imaginary part

	Returns:
		Hmean:   ndarray, shape (N,)
				 best estimate of complex frequency response values
		Hcov:    ndarray, shape (2N,2N)
				 covariance matrix [ [u(real,real), u(real,imag)], [u(imag,real), u(imag,imag)] ]

	"""

	runs = 10000
	SMC = S + np.random.randn(runs)*uS
	dMC = d + np.random.randn(runs)*ud
	fMC = f0+ np.random.randn(runs)*uf0

	HMC = FreqResp(SMC,dMC,fMC,f)

	return np.mean(HMC,dtype=complex,axis=1), np.cov(np.vstack((np.real(HMC),np.imag(HMC))),rowvar=1)


def unc_absphase(S, d, f0, uS, ud, uf0, f):
	"""Propagation of uncertainty from physical parameters to real and imaginary
	part of system's transfer function using GUM S2 Monte Carlo.

	Parameters:
		S:    float
			  static gain
		d:    float
			  damping
		f0:   float
			  resonance frequency
		uS:   float
			  uncertainty associated with static gain
		ud:   float
			  uncertainty associated with damping
		uf0:  float
			  uncertainty associated with resonance frequency
		f:    ndarray, shape (N,)
			  frequency values at which to calculate amplitue and phase

		Hmean:   ndarray, shape (N,)
				 best estimate of complex frequency response values
		Hcov:    ndarray, shape (2N,2N)
				 covariance matrix [ [u(abs,abs), u(abs,phase)], [u(phase,abs), u(phase,phase)] ]

	"""

	runs = 10000
	SMC = S + np.random.randn(runs)*uS
	dMC = d + np.random.randn(runs)*ud
	fMC = f0+ np.random.randn(runs)*uf0

	HMC = FreqResp(SMC,dMC,fMC,f)

	return np.mean(HMC,dtype=complex,axis=1), np.cov(np.vstack((np.abs(HMC),ua(HMC))),rowvar=1)

