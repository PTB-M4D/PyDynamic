# -*- coding: utf-8 -*-
"""

The propagation of uncertainties via the FIR and IIR formulae alone does not
enable the derivation of credible intervals, because the underlying distribution
remains unknown. The GUM-S2 Monte Carlo method provides a reference method for the
calculation of uncertainties for such cases.

This module implements Monte Carlo methods for the propagation of uncertainties for digital filtering.

"""

# TODO: Implement updating Monte Carlo method

import numpy as np
import scipy as sp
from numpy import matrix
import sys
from scipy.signal import lfilter

from ..misc.tools import zerom
from ..misc.filterstuff import isstable


def MC(x,noise_std,b,a,Uab,runs=1000,blow=None,alow=None,return_samples=False,shift=0,verbose=True):
	"""Standard Monte Carlo method

	Monte Carlo based propagation of uncertainties for a digital filter (b,a)
	with uncertainty matrix
	:math:`U_{\theta}` for :math:`\theta=(a_1,\ldots,a_{N_a},b_0,\ldots,b_{N_b})^T`

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

	Y = np.zeros((runs,len(x)))
	theta = np.hstack((a[1:],b))
	Theta = np.random.multivariate_normal(theta,Uab,runs)

	unst_count = 0
	st_inds  = list()
	if verbose:
		sys.stdout.write('MC progress: ')
	for k in range(runs):
		xn = x + np.random.randn(len(x))*noise_std
		if not blow is None:
			if alow is None:
				alow = 1.0
			xn = lfilter(blow,alow,xn)
		bb = Theta[k,Na-1:]
		aa = np.hstack((1.0, Theta[k,:Na-1]))
		if isstable(bb,aa):
			Y[k,:] = lfilter(bb,aa,xn)
			st_inds.append(k)
		else:
			unst_count += 1
		if np.mod(k, 0.1*runs) == 0 and verbose:
			sys.stdout.write(' %d%%' % (np.round(100.0*k/runs)))
	if verbose:
		sys.stdout.write(" 100%\n")

	if unst_count > 0:
		print("In %d Monte Carlo %d filters have been unstable" % (runs,unst_count))
		print("These results will not be considered for calculation of mean and std")
		print("However, if return_samples is 'True' then ALL samples are returned.")

	Y = np.roll(Y,int(shift),axis=1)

	if return_samples:
		return Y
	else:
		y = np.mean(Y[st_inds,:],axis=0)
		uy= np.std(Y[st_inds,:],axis=0)
		return y, uy



def SMC(x,noise_std,b,a,Uab,runs=1000,Perc=None,blow=None,alow=None,shift=0,\
			return_samples=False,phi=None,theta=None,Delta=0.0):
	"""Sequential Monte Carlo method

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

	if isinstance(a,np.ndarray):
		Na = len(a)-1
	else:
		Na = 0
	if isinstance(b,np.ndarray):
		Nb = len(b)-1
	else:
		Nb = 0

	if isinstance(theta,np.ndarray) or isinstance(theta,float):
		if isinstance(theta,float):
			W = zerom((runs,1))
		else:
			W = zerom((runs,len(theta)))
			theta = matrix(theta[:])
	else:
		MA = False

	if isinstance(phi,np.ndarray) or isinstance(phi,float):
		AR = True
		if isinstance(phi,float):
			E = zerom((runs,1))
		else:
			E = zerom((runs,len(phi)))
			phi = matrix(phi[:])
	else:
		AR = False



	if isinstance(blow,np.ndarray):
		X = zerom((runs,len(blow)))
		blow = matrix(blow[:])
	else:
		X = zerom((runs,1))

	if isinstance(alow,np.ndarray):
		Xl = zerom((runs,len(alow)-1))
		alow = matrix(alow[:])
	else:
		Xl = zerom((runs,1))



	if Na==0:
		coefs = b
	else:
		coefs = np.hstack((a[1:],b))

	Coefs = matrix(np.random.multivariate_normal(coefs,Uab,runs))

	b0 = Coefs[:,Na]

	if Na>0: # filter is IIR
		A = Coefs[:,:Na]
		if Nb>Na:
			A = np.hstack((A,zerom((runs,Nb-Na))))
	else: # filter is FIR -> zero state equations
		A = zerom((runs,Nb))

	c = Coefs[:,Na+1:] - np.multiply(np.tile(b0,(1,Nb)),A)
	States = zerom(np.shape(A))

	calcP = False
	if not Perc is None:
		calcP = True
		P = np.zeros((len(Perc),len(x)))

	y = np.zeros_like(x)
	Uy= np.zeros_like(x)


	sys.stdout.write("SMC progress: ")
	for k in range(len(x)):

		w  = np.random.randn(runs)*noise_std
		if AR and MA:
			E  = np.hstack( ( E*phi + W*theta + w, E[:-1]) )
			W  = np.hstack( (w, W[:-1] ) )
		elif AR:
			E  = np.hstack( (E*phi + w, E[:-1]) )
		elif MA:
			E  = W*theta + w
			W  = np.hstack( (w, W[:-1] ) )
		else:
			w  = np.random.randn(runs)*noise_std
			E  = matrix(w).T

		X  = np.hstack( (x[k] + E,  X[:,:-1]) )
		if isinstance(alow,np.matrix):
			Xl = np.hstack( ( X*blow.T - Xl[:,:len(alow)]*alow[1:], Xl[:,:-1] ) )
		elif isinstance(blow,np.matrix):
			Xl = X*blow.T
		else:
			Xl = X


		Y = np.sum(np.multiply(c,States),axis=1) + np.multiply(b0,Xl[:,0]) + (np.random.rand(runs,1)*2*Delta - Delta)
		Z = -np.sum(np.multiply(A,States),axis=1) + Xl[:,0]
		States = np.hstack((Z, States[:,:-1]))

		y[k] = np.mean(Y)
		Uy[k]= np.std(Y)
		if calcP:
			P[:,k] = sp.stats.mstats.mquantiles(np.asarray(Y),prob=Perc)

		if np.mod(k, np.round(0.1*len(x))) == 0:
			sys.stdout.write(' %d%%' % (np.round(100.0*k/len(x))))

	sys.stdout.write(" 100%\n")


	y = np.roll(y,int(shift))
	Uy= np.roll(Uy,int(shift))


	if calcP:
		P = np.roll(P,int(shift),axis=1)
		return y, Uy, P
	else:
		return y, Uy
