# -*- coding: utf-8 -*-

import numpy as np
import statsmodels.api as sm
import scipy as sp

class DynamicModel(object):
	"""
	This is the base class for the nodal load observer state evolution model.
	"""
	parameters = dict([])
	def __init__(self,dim):
		self.dim = dim
		self.x0 = np.zeros(dim)
		self.P0 = np.zeros((dim,dim))

	def forecast_state(self,x=None):
		"""
		Calculate one step ahead prediction for the chosen model
		:param x: current value
		:return: predicted value
		"""
		raise NotImplementedError

	def forecast_unc(self,P=None):
		"""
		Calculate one step ahead prediction of error covariance
		:param P: current covariance
		:return: predicted covariance
		"""
		raise NotImplementedError

	def fit_model(self, data):
		"""
		Fit model parameters A to the data
		"""
		raise NotImplementedError
	def adjust_Dnm(self, Dnm):
		"""
		Some models (like AR2) require an adjustment of the mapping matrix Dnm
		"""
		return Dnm



class SimpleModel(DynamicModel):
	"""
	The generic structure of the simple model is given as
		x(k+1) = alpha * I_n x(k) + noise
	with I_n the identity matrix of dimension n
	"""
	def __init__(self,dim,alpha=0.95, q = 10):
		super(SimpleModel,self).__init__(dim)
		self.alpha = alpha
		self.q = q
		self.parameters["alpha"] = alpha
		self.parameters["noise_var"] = q

	def forecast_state(self,x=None):
		# extended Kalman filter state prediction step
		if not isinstance(x,np.ndarray):
			x = self.x0
		return self.alpha*x

	def forecast_unc(self,P=None):
		# extended Kalman filter prediction step for error covariance P
		if not isinstance(P,np.ndarray):
			P = self.P0
		return self.alpha**2 * P + self.q*np.eye(self.dim)

	def return_pars(self):
		return self.alpha*np.eye(self.dim), self.q*np.eye(self.dim)



class AR2Model_single(DynamicModel):
	"""
	The AR(2) model is given as
		(x(k+1),x(k)) = (phi1,phi2; 1,0) (x(k),x(k-1)) + noise
	and is considered to be fitted to a one dimensional time series.
	Hence, the same model is used for all pseudo-measurements.
	"""

	def __init__(self,dim,phi1=None,phi2=None,noise=None):
		super(AR2Model_single,self).__init__(2*dim) # using 2*dim to account for AR2

		if isinstance(phi1,float):
			self.parameters["phi"] = np.r_[phi1,phi2]
			self.setA()
		if isinstance(noise,float):
			self.setQ(noise)

	def adjust_Dnm(self, Dnm):
		Dtilde = np.zeros((Dnm.shape[1],self.dim))
		for i in range(Dnm.shape[1]):
			Dtilde[i,2*i] = 1.0
		return np.dot(Dnm,Dtilde)

	def setQ(self,noise):
		self.Q = sp.linalg.block_diag(noise*np.eye(self.dim/2),
									  np.zeros((self.dim/2,self.dim/2)))

	def setA(self):
		A11 = self.parameters["phi"][0]*np.eye(self.dim/2)
		A12 = self.parameters["phi"][0]*np.eye(self.dim/2)
		A21 = np.eye(self.dim/2)
		A22 = np.zeros_like(A21)
		self.A = np.vstack((np.hstack((A11,A12)),
							np.hstack((A21,A22)))) # (2*(n_k-1),2*(n_k-1))

	def fit_model(self, data):
		arma_mod20 = sm.tsa.ARMA(data, (2,0)).fit()
		self.parameters["constant"] = arma_mod20.params[0]
		self.parameters["phi"] = arma_mod20.params[1:]
		self.setA()

	def forecast_state(self,x=None):
		if not isinstance(x,np.ndarray):
			x = self.x0
		if "phi" in self.parameters:
			return np.dot(self.A, x)
		else:
			raise NotImplementedError("The model parameters have not been determined yet.")

	def forecast_unc(self,P=None):
		if not isinstance(P,np.ndarray):
			P = self.P0
		if "phi" in self.parameters:
			return np.dot(self.A,np.dot(P,self.A.T)) + self.Q
		else:
			raise NotImplementedError("The model parameters have not been determined yet.")

