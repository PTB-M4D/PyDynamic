# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from numpy import matrix
from scipy.signal import tf2ss

if __name__=="uncertainty.propagate_IIR":	 #module is imported from within package
	from misc.tools import zerom
else:
	from ..misc.tools import zerom


def IIR_uncFilter(x,noise,b,a,Uab):
	"""
	Uncertainty propagation for the signal x and the uncertain IIR filter (b,a)

	:param x: filter input signal
	:param noise: signal noise standard deviation
	:param b: filter numerator coefficients
	:param a: filter denominator coefficients
	:param Uba: 2D matrix of covariance (mutual uncertainties) for (a[1:],b[0:])

	:returns y: filter output signal
	:returns Uy: uncertainty associated with y

	Implementation of the IIR formula for uncertainty propagation:

	A. Link and C. Elster
	Uncertainty evaluation for IIR filtering using a state-space approach
	Meas. Sci. Technol. vol. 20, 2009, [DOI](http://dx.doi.org/10.1088/0957-0233/20/5/055104)
	
	"""

	if not isinstance(noise,np.ndarray):
		noise = noise*np.ones(np.shape(x))

	p = len(a)-1
	if not len(b)==len(a):
		b = np.hstack((b,np.zeros((len(a)-len(b),))))

	# from discrete-time transfer function to state space representation
	[A,bs,c,b0] = tf2ss(b,a)

	A = matrix(A); bs = matrix(bs); c = matrix(c);

	phi = zerom((2*p+1,1))
	dz  = zerom((p,p)); dz1 = zerom((p,p))
	z   = zerom((p,1))
	P   = zerom((p,p))

	y = np.zeros((len(x),))
	Uy= np.zeros((len(x),))

	Aabl = np.zeros((p,p,p))
	for k in range(p):
		Aabl[0,k,k] = -1


	for n in range(len(y)):
		for k in range(p): # derivative w.r.t. a_1,...,a_p
			dz1[:,k]= A*dz[:,k] + np.squeeze(Aabl[:,:,k])*z
			phi[k] = c*dz[:,k] - b0*z[k]
		phi[p+1] = -matrix(a[1:])*z + x[n] # derivative w.r.t. b_0
		for k in range(p+2,2*p+1): # derivative w.r.t. b_1,...,b_p
			phi[k] = z[k-(p+1)]
		P = A*P*A.T + noise[n]**2*(bs*bs.T)
		y[n] = c*z + b0*x[n]
		Uy[n]= phi.T*Uab*phi + c*P*c.T + b[0]**2*noise[n]**2
		# update of the state equations
		z = A*z + bs*x[n]
		dz = dz1

	Uy = np.sqrt(np.abs(Uy))

	return y, Uy


