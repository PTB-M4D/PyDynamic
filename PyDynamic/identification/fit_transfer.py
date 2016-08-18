# -*- coding: utf-8 -*-
"""

Collection of methods for the identification of transfer function models

"""
import numpy as np

def fit_sos(f, H, UH=None, unc_type="realimag", weighting=None, MCruns = None, scaling = 1e-3):
	"""Fit second-order model (spring-damper model) with parameters S0, delta and f0
	to complex-valued frequency response with uncertainty associated with amplitude and phase
	or associated with real and imaginary parts

	Parameters
	----------
		f: np.ndarray
			vector of frequencies
		H: np.ndarray
			complex-valued frequency response values at frequencies f
		UH: np.ndarray
			uncertainties associated either with amplitude and phase of H or real and imaginary parts
			When UH is one-dimensional, it is assumed to contain standard uncertainties; otherwise it
			is taken as covariance matrix. When UH is not specified no uncertainties assoc. with the fit are calculated.
		unc_type: str
			Determining type of uncertainty. "ampphase" for uncertainties associated with amplitude and
			phase. "realimag" for uncertainties associated with real and imaginary parts
		weighting: str or array
			Type of weighting (None, 'diag', 'cov') or array of weights (length two times of f)
		MCruns: int
			Number of Monte Carlo trials for propagation of uncertainties. When MCruns is 'None', matrix multiplication
			is used for the propagation of uncertainties. However, in some cases this can cause trouble.
		scaling: float
			scaling of least-squares design matrix for improved fit quality

	Returns
	-------
		p: np.ndarray
			vector of estimated model parameters [S0, delta, f0]
		Up: np.ndarray
			covariance associated with parameter estimate
	"""

	assert(len(f)==len(H))
	assert(H.dtype==complex)

	if isinstance(UH, np.ndarray):
		assert(UH.shape[0]==2*len(H))
		if len(UH.shape)==2:
			assert(UH.shape[0]==UH.shape[1])

		# propagate to real and imaginary parts of reciprocal using Monte Carlo
		if isinstance(MCruns, int) or isinstance(MCruns, float):
			runs = int(MCruns)
		else:
			runs = 10000
		if unc_type=="ampphase":
			if len(UH.shape)==1:
				Habs = np.tile(np.abs(H), (runs, 1)) + np.random.randn(runs, len(f)) * np.tile( UH[:len(f)], (runs, 1))
				Hang = np.tile(np.angle(H),(runs,1)) + np.random.randn(runs, len(f)) * np.tile( UH[len(f):], (runs, 1))
				HMC = Habs * np.exp( 1j * Hang)
			else:
				HMC = np.random.multivariate_normal(np.r_[np.abs(H), np.angle(H)], UH, runs)
		elif unc_type=="realimag":
			if len(UH.shape)==1:
				HR = np.tile(np.real(H), (runs, 1)) + np.random.randn(runs, len(f)) * np.tile( UH[:len(f)], (runs, 1))
				HI = np.tile(np.imag(H), (runs, 1)) + np.random.randn(runs, len(f)) * np.tile( UH[len(f):], (runs, 1))
				HMC = HR + 1j*HI
			else:
				HRI = np.random.multivariate_normal(np.r_[np.real(H), np.imag(H)], UH, runs)
				HMC = HRI[:,:len(f)] + 1j*HRI[:,len(f):]
		else:
			raise ValueError("Wrong type of uncertainty")

		iRI = np.c_[np.real(1/HMC), np.imag(1/HMC)]
		iURI= np.cov(iRI, rowvar = False)


	if isinstance(weighting, str):
		if weighting == "diag":
			W = np.diag(np.diag(iURI))
		if weighting == "cov":
			W = iURI
		else:
			print("Warning: Specified wrong type of weighting.")
			W = np.eye(2*len(f))
	elif isinstance(weighting, np.ndarray):
		assert(len(weighting)==2*len(f))
		W = np.diag(weighting)
	else:
		W = np.eye(2*len(f))

	if isinstance(UH, np.ndarray):
		if isinstance(MCruns, int) or isinstance(MCruns, float):
			runs = int(MCruns)
			MU = np.zeros((runs, 3))
			for k in range(runs):
				iri = iRI[k,:]
				n = len(f)
				om = 2 * np.pi * f * scaling
				E = np.c_[np.ones(n), 2j * om, - om**2]
				X = np.r_[np.real(E), np.imag(E)]

				XVX = (X.T).dot(np.linalg.solve(W, X))
				XVy = (X.T).dot(np.linalg.solve(W, iri))

				MU[k,:] = np.linalg.solve(XVX, XVy)
			MU[:,1] *= scaling
			MU[:,2] *= scaling**2

			# calculate S0, delta and f0
			PARS = np.c_[1/MU[:,0], MU[:,1]/np.sqrt(np.abs(MU[:,0]*MU[:,2])), np.sqrt(np.abs(MU[:,0]/MU[:,2]))/2/np.pi]

			pars = PARS.mean(axis=0)
			Upars= np.cov(PARS, rowvar = False)
		else:
			iri = np.r_[np.real(1 / H), np.imag(1 / H)]
			n = len(f)
			om = 2 * np.pi * f
			E = np.c_[np.ones(n), 2j * om, - om ** 2]
			X = np.r_[np.real(E), np.imag(E)]

			XVX = (X.T).dot(np.linalg.solve(W, X))
			XVy = (X.T).dot(np.linalg.solve(W, iri))

			mu = np.linalg.solve(XVX, XVy)
			Umu= np.linalg.solve( XVX, (X.T).dot(np.linalg.solve(W, iURI))  )

			pars = np.r_[1 / mu[0], mu[1] / np.sqrt(np.abs(mu[0] * mu[2])), np.sqrt(np.abs(mu[0] / mu[2])) / 2 / np.pi]

		return pars, Upars
	else:
		iri = np.r_[np.real(1/H), np.imag(1/H)]
		n = len(f)
		om = 2 * np.pi * f * scaling
		E = np.c_[np.ones(n), 2j * om, - om**2]
		X = np.r_[np.real(E), np.imag(E)]

		XVX = (X.T).dot(np.linalg.solve(W, X))
		XVy = (X.T).dot(np.linalg.solve(W, iri))

		mu = np.linalg.solve(XVX, XVy)
		mu[1] *= scaling
		mu[2] *= scaling**2
		pars  = np.r_[1/mu[0], mu[1]/np.sqrt(np.abs(mu[0]*mu[2])), np.sqrt(np.abs(mu[0]/mu[2]))/2/np.pi]
		return pars




