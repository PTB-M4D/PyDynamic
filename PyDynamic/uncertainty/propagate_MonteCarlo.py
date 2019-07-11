# -*- coding: utf-8 -*-
"""

The propagation of uncertainties via the FIR and IIR formulae alone does not
enable the derivation of credible intervals, because the underlying distribution
remains unknown. The GUM-S2 Monte Carlo method provides a reference method for the
calculation of uncertainties for such cases.

This module implements Monte Carlo methods for the propagation of uncertainties for digital filtering.

This module contains the following functions:
* MC: Standard Monte Carlo method for application of digital filter
* SMC: Sequential Monte Carlo method with reduced computer memory requirements

"""

import sys

import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.signal import lfilter

from ..misc.filterstuff import isstable
from ..misc.tools import zerom

__all__ = ["MC", "SMC"]

class Normal_ZeroCorr():
	"""
	Multivariate normal distribution with zero correlation
	"""
	def __init__(self, mean=None, cov=None):
		"""
		Parameters
		----------
			loc: np.ndarray, optional
				mean values, default is zero
			scale: np.ndarray, optional
				standard deviations for the elements in loc, default is zero
		"""
		if isinstance(mean, np.ndarray):
			self.mean = mean
			if isinstance(cov, np.ndarray):
				assert (len(cov)==len(mean))
				self.std = cov
			elif isinstance(cov, float):
				self.std = cov * np.ones_like(mean)
			else:
				self.std = np.zeros_like(mean)
		elif isinstance(cov, np.ndarray):
			self.std = cov
			self.mean = np.zeros_like(cov)

	def rvs(self, size=1):
		# This function mimics the behavior of the scipy stats package
		return np.tile(self.mean, (size, 1)) + np.random.randn(size, len(self.mean))*np.tile(self.std, (size, 1))



def MC(x,Ux,b,a,Uab,runs=1000,blow=None,alow=None,return_samples=False,shift=0,verbose=True):
	r"""Standard Monte Carlo method

	Monte Carlo based propagation of uncertainties for a digital filter (b,a)
	with uncertainty matrix
	:math:`U_{\theta}` for :math:`\theta=(a_1,\ldots,a_{N_a},b_0,\ldots,b_{N_b})^T`

	Parameters
	----------
		x: np.ndarray
			filter input signal
		Ux: float or np.ndarray
			standard deviation of signal noise (float), point-wise standard uncertainties or covariance matrix associated with x
		b: np.ndarray
			filter numerator coefficients
		a: np.ndarray
			filter denominator coefficients
		Uab: np.ndarray
			uncertainty matrix :math:`U_\theta`
		runs: int,optional
			number of Monte Carlo runs
		return_samples: bool, optional
			whether samples or mean and std are returned

	If 'return_sampes' is false, the method returns:

	Returns
	-------
		y: np.ndarray
			filter output signal
		Uy: np.ndarray
			uncertainty associated with

	Other wise the method returns

	Returns
	-------
		Y: np.ndarray
			array of Monte Carlo results
	
	References
	----------
		* Eichstädt, Link, Harris and Elster [Eichst2012]_
	"""

	Na = len(a)
	runs = int(runs)

	Y = np.zeros((runs,len(x)))		# set up matrix of MC results
	theta = np.hstack((a[1:],b))	# create the parameter vector from the filter coefficients
	Theta = np.random.multivariate_normal(theta,Uab,runs)	# Theta is small and thus we can draw the full matrix now.
	if isinstance(Ux, np.ndarray):
		if len(Ux.shape)==1:
			dist = Normal_ZeroCorr(loc=x, scale=Ux) 	# non-iid noise w/o correlation
		else:
			dist = stats.multivariate_normal(x, Ux)		# colored noise
	elif isinstance(Ux, float):
		dist = Normal_ZeroCorr(mean=x, cov=Ux)			# iid noise
	else:
		raise NotImplementedError("The supplied type of uncertainty is not implemented")

	unst_count = 0			# Count how often in the MC runs the IIR filter is unstable.
	st_inds  = list()
	if verbose:
		sys.stdout.write('MC progress: ')
	for k in range(runs):
		xn = dist.rvs()		# draw filter input signal
		if not blow is None:
			if alow is None:
				alow = 1.0  # FIR low-pass filter
			xn = lfilter(blow,alow,xn)	# low-pass filtered input signal
		bb = Theta[k,Na-1:]
		aa = np.hstack((1.0, Theta[k,:Na-1]))
		if isstable(bb,aa):
			Y[k,:] = lfilter(bb,aa,xn)
			st_inds.append(k)
		else:
			unst_count += 1		# don't apply the IIR filter if it's unstable
		if np.mod(k, 0.1*runs) == 0 and verbose:
			sys.stdout.write(' %d%%' % (np.round(100.0*k/runs)))
	if verbose:
		sys.stdout.write(" 100%\n")

	if unst_count > 0:
		print("In %d Monte Carlo %d filters have been unstable" % (runs,unst_count))
		print("These results will not be considered for calculation of mean and std")
		print("However, if return_samples is 'True' then ALL samples are returned.")

	Y = np.roll(Y,int(shift),axis=1)		# correct for the (known) sample delay

	if return_samples:
		return Y
	else:
		y = np.mean(Y[st_inds,:],axis=0)
		uy= np.cov(Y[st_inds,:],rowvar=0)
		return y, uy



def SMC(x,noise_std,b,a,Uab=None,runs=1000,Perc=None,blow=None,alow=None,shift=0,
			return_samples=False,phi=None,theta=None,Delta=0.0):
	r"""Sequential Monte Carlo method

	Sequential Monte Carlo propagation for a digital filter (b,a) with uncertainty
	matrix :math:`U_{\theta}` for :math:`\theta=(a_1,\ldots,a_{N_a},b_0,\ldots,b_{N_b})^T`

	Parameters
	----------
		x: np.ndarray
			filter input signal
		noise_std: float
			standard deviation of signal noise
		b: np.ndarray
			filter numerator coefficients
		a: np.ndarray
			filter denominator coefficients
		Uab: np.ndarray
			uncertainty matrix :math:`U_\theta`
		runs: int, optional
			number of Monte Carlo runs
		Perc: list, optional
			list of percentiles for quantile calculation
		blow: np.ndarray
			optional low-pass filter numerator coefficients
		alow: np.ndarray
			optional low-pass filter denominator coefficients
		shift: int
			integer for time delay of output signals
		return_samples: bool, otpional
			whether to return y and Uy or the matrix Y of MC results
		phi, theta: np.ndarray, optional
			parameters for AR(MA) noise model
			::math:`\epsilon(n)  = \sum_k \phi_k\epsilon(n-k) + \sum_k \theta_k w(n-k) + w(n)`
			with ::math:`w(n)\sim N(0,noise_std^2)`
		Delta: float,optional
		 	upper bound on systematic error of the filter

	If return_samples is False:

	Returns
	-------
		y: np.ndarray
			filter output signal (Monte Carlo mean)
		Uy: np.ndarray
			uncertainties associated with y (Monte Carlo point-wise std)
		Quant: np.ndarray
			quantiles corresponding to percentiles Perc (if not None) at

	Otherwise:

	Returns
	-------
		Y: np.ndarray
			array of all Monte Carlo results

	References
	----------
		* Eichstädt, Link, Harris, Elster [Eichst2012]_
	"""

	runs = int(runs)

	if isinstance(a,np.ndarray):	# filter order denominator
		Na = len(a)-1
	else:
		Na = 0
	if isinstance(b,np.ndarray):	# filter order numerator
		Nb = len(b)-1
	else:
		Nb = 0

	if isinstance(theta,np.ndarray) or isinstance(theta,float):		# initialise noise matrix corresponding to ARMA noise model
		if isinstance(theta,float):
			W = np.zeros((runs,1))
		else:
			W = np.zeros((runs,len(theta)))
	else:
		MA = False		# no moving average part in noise process

	if isinstance(phi,np.ndarray) or isinstance(phi,float):			# Initialise for autoregressive part of noise process
		AR = True
		if isinstance(phi,float):
			E = np.zeros((runs,1))
		else:
			E = np.zeros((runs,len(phi)))
	else:
		AR = False		# no autoregresssive part in noise process



	if isinstance(blow,np.ndarray):
		X = np.zeros((runs,len(blow)))		# initialise matrix of low-pass filtered input signal
	else:
		X = np.zeros(runs)

	if isinstance(alow,np.ndarray):
		Xl = np.zeros((runs,len(alow)-1))
	else:
		Xl = np.zeros((runs,1))



	if Na==0:		# only FIR filter
		coefs = b
	else:
		coefs = np.hstack((a[1:],b))

	if isinstance(Uab, np.ndarray):			# Monte Carlo draw for filter coefficients
		Coefs = np.random.multivariate_normal(coefs, Uab, runs)
	else:
		Coefs = np.tile(coefs, (runs, 1))

	b0 = Coefs[:,Na]

	if Na>0:		# filter is IIR
		A = Coefs[:,:Na]
		if Nb>Na:
			A = np.hstack((A,zerom((runs,Nb-Na))))
	else:			# filter is FIR -> zero state equations
		A = np.zeros((runs,Nb))

	c = Coefs[:,Na+1:] - np.multiply(np.tile(b0[:, np.newaxis],(1,Nb)),A)	# Fixed part of state-space model
	States = np.zeros(np.shape(A))		# initialise matrix of states

	calcP = False			# by default no percentiles requested
	if not Perc is None:	# percentiles requested
		calcP = True
		P = np.zeros((len(Perc),len(x)))

	y = np.zeros_like(x)	# initialise outputs
	Uy= np.zeros_like(x)	# initialise vector of point-wise uncertainties (no correlation)


	print("Sequential Monte Carlo progress", end="")	# start of the actual MC part
	for k in range(len(x)):

		w  = np.random.randn(runs)*noise_std		# noise process draw
		if AR and MA:
			E  = np.hstack( ( E.dot(phi) + W.dot(theta) + w, E[:-1]) )
			W  = np.hstack( (w, W[:-1] ) )
		elif AR:
			E  = np.hstack( (E.dot(phi) + w, E[:-1]) )
		elif MA:
			E  = W.dot(theta) + w
			W  = np.hstack( (w, W[:-1] ) )
		else:
			w  = np.random.randn(runs,1)*noise_std
			E  = w


		if isinstance(alow,np.ndarray):				# apply low-pass filter
			X = np.hstack((x[k] + E, X[:, :-1]))
			Xl = np.hstack( ( X.dot(blow.T) - Xl[:,:len(alow)].dot(alow[1:]), Xl[:,:-1] ) )
		elif isinstance(blow,np.ndarray):
			X = np.hstack((x[k] + E, X[:, :-1]))
			Xl = X.dot(blow)
		else:
			Xl = x[k] + E
		if len(Xl.shape)==1:
			Xl = Xl[:,np.newaxis]		# prepare for easier calculations
		Y = np.sum(np.multiply(c,States),axis=1) + np.multiply(b0,Xl[:,0]) + (np.random.rand(runs)*2*Delta - Delta) 	# state-space system output
		Z = -np.sum(np.multiply(A,States),axis=1) + Xl[:,0]			# calculate state updates
		States = np.hstack((Z[:,np.newaxis], States[:,:-1]))		# store state updates and remove old ones

		y[k] = np.mean(Y)		# point-wise best estimate
		Uy[k]= np.std(Y)		# point-wise standard uncertainties
		if calcP:
			P[:,k] = sp.stats.mstats.mquantiles(np.asarray(Y),prob=Perc)	# percentiles if requested

		if np.mod(k, np.round(0.1*len(x))) == 0:
			print(' %d%%' % (np.round(100.0*k/len(x))), end="")

	print(" 100%")


	y = np.roll(y,int(shift))	# correct for (known) delay
	Uy= np.roll(Uy,int(shift))


	if calcP:
		P = np.roll(P,int(shift),axis=1)
		return y, Uy, P
	else:
		return y, Uy
