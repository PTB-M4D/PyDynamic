# -*- coding: utf-8 -*-
"""

The propagation of uncertainties via the FIR and IIR formulae alone does not
enable the derivation of credible intervals, because the underlying distribution
remains unknown. To this end, the GUM-S2 Monte Carlo is carried out.

Therefore, this module contains several approaches to a Monte Carlo based
propagation of uncertainties.

"""

import numpy as np
import scipy as sp
from numpy import matrix
import sys
from scipy.signal import lfilter

if __name__=="uncertainty.propagate_MonteCarlo":	 #module is imported from within package
	from misc.tools import zerom
	from misc.filterstuff import isstable
else:
	from ..misc.tools import zerom
	from ..misc.filterstuff import isstable


def MC(x,noise_std,b,a,Uab,runs=1000,blow=None,alow=None,return_samples=False,shift=0,verbose=True):
	"""
	Monte Carlo based propagation of uncertainties for a digital filter (b,a)
	with uncertainty matrix
	:math:`U_{\theta}` for :math:`\theta=(a_1,\ldots,a_{N_a},b_0,\ldots,b_{N_b})^T`

	:param x: filter input signal
	:param noise_std: standard deviation of signal noise
	:param b: filter numerator coefficients
	:param a: filter denominator coefficients
	:param Uab: uncertainty matrix :math:`U_\theta`
	:param runs: number of Monte Carlo runs
	:param return_samples: boolean whether samples or mean and std are returned

	If return_samples is set to False:
	:returns y, Uy: filter output and associated uncertainty
	Otherwise:
	:returns Y: matrix of Monte Carlo results
	
	Application of Monte Carlo method from
	
	S. Eichstädt, A. Link, P. M. Harris und C. Elster 
	Efficient implementation of a Monte Carlo method for uncertainty evaluation in dynamic measurements. 
	Metrologia, 49(3), 401, 2012. [DOI](http://dx.doi.org/10.1088/0026-1394/49/3/401)

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
	"""
	Sequential Monte Carlo propagation for a digital filter (b,a) with uncertainty
	matrix :math:`U_{\theta}` for :math:`\theta=(a_1,\ldots,a_{N_a},b_0,\ldots,b_{N_b})^T`

	:param x: filter input signal
	:param noise_std: standard deviation of signal noise
	:param b: filter numerator coefficients
	:param a: filter denominator coefficients
	:param Uab: uncertainty matrix :math:`U_\theta`
	:param runs: number of Monte Carlo runs
	:param Perc: (default None) optional ndarray of percentiles for quantile calculation
	:param blow, alow: optional lowpass filter coefficients (blow,alow)
	:param shift: optional integer for time shift of output signals
	:param return_samples: boolean whether to return y and Uy or the matrix Y of MC results
	:param phi,theta: parameters for AR(MA) noise model
		::math:`\epsilon(n)  = \sum_k \phi_k\epsilon(n-k) + \sum_k \theta_k w(n-k) + w(n)`
		with ::math:`w(n)\sim N(0,noise_std^2)`
	:param Delta: (float) upper bound on systematic error, default is 0.0

	If return_samples is False (default):
	:returns y: filter output signal (Monte Carlo mean)
	:returns Uy: uncertainties associated with y (Monte Carlo point-wise std)
	:returns Quant: ndarray of quantiles corresponding to percentiles Perc (if not None) at
	each time instant
	Otherwise:
	:returns Y: matrix of Monte Carlo results

	Application of Monte Carlo method from
	
	S. Eichstädt, A. Link, P. M. Harris und C. Elster 
	Efficient implementation of a Monte Carlo method for uncertainty evaluation in dynamic measurements. 
	Metrologia, 49(3), 401, 2012. [DOI](http://dx.doi.org/10.1088/0026-1394/49/3/401)

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
