# -*- coding: utf-8 -*-

"""
Propagation of uncertainty for application of FFT, inverse FFT and deconvolution in the frequency domain
Methods originally developed for GUM2DFT
	S. Eichstädt und V. Wilkens
	GUM2DFT — a software tool for uncertainty evaluation of transient signals in the frequency domain.
	Measurement Science and Technology, 27(5), 055001, 2016. [DOI: 10.1088/0957-0233/27/5/055001]

"""

import numpy as np
from scipy import sparse

def apply_window(x,Ux,window):
	"""Apply a time domain window to the signal x of equal length and propagate uncertainties

	Args:
		x: vector of time domain signal values
		Ux: covariance matrix associated with x or noise variance as float
		window: vector of time domain window (same length as x)
	Returns:
		xw,Uxw
	"""
	assert(len(x)==len(window))
	if not isinstance(Ux,float):
		assert(Ux.shape[0]==Ux.shape[1] and Ux.shape[0]==len(x))
	xw = x.copy()*window
	if isinstance(Ux,float):
		Uxw = Ux*window**2
	else:
		Uxw = prod(window,prod(Ux,window))
	return xw,Uxw

def prod(A,B):
	"""Calculate the matrix-vector product, or vector-matrix product
	that corresponds to diag(A)*B or A*diag(B), respectively; depending
	on which of A,B is the matrix and which the vector.
	"""
	if len(A.shape)==1 and len(B.shape)==2:  # A is the vector and B the matrix
		C = np.zeros_like(B)
		for k in range(C.shape[0]):
			C[k,:] = A[k]*B[k,:]
		return C
	elif len(A.shape)==2 and len(B.shape)==1: # A is the matrix and B the vector
		C = np.zeros_like(A)
		for k in range(C.shape[1]):
			C[:,k] = A[:,k]*B[k]
		return C
	else:
		raise ValueError("Wrong dimension of inputs")

def matprod(M,V,W):
	"""Calculate the matrix-matrix-matrix product (V1,V2)M(W1,W2) for V=(V1,V2)
	and W=(W1,W2). M can be sparse, one-dimensional or a full (quadratic) matrix.
	"""
	if len(M.shape)==2:
		assert(M.shape[0]==M.shape[1])
	assert(M.shape[0]==V.shape[0])
	assert(V.shape==W.shape)
	N = V.shape[0]/2
	v1 = V[:N]; v2 = V[N:]
	w1 = W[:N]; w2 = W[N:]
	if isinstance(M,sparse.dia_matrix):
		nrows = M.shape[0]
		offset= M.offsets
		diags = M.data
		A = diags[0][:N]
		B = diags[1][offset[1]:nrows+offset[1]]
		D = diags[0][N:]
		return np.diag(v1*A*w1 + v2*B*w1 + v1*B*w2 + v2*D*w2)
	elif len(M.shape)==1:
		A = M[:N]
		D = M[N:]
		return np.diag(v1*A*w1 + v2*D*w2)
	else:
		A = M[:N,:N]
		B = M[:N,N:]
		D = M[N:,N:]
		return prod(v1,prod(A,w1)) + prod(v2,prod(B.T,w1)) + prod(v1,prod(B,w2)) + prod(v2,prod(D,w2))




def GUM_DFT(x,Ux,N=None,window=None,CxCos=None,CxSin=None,returnC=False):
	"""Calculation of DFT of time domain signal x and propagation of uncertainty U_x
	associated with the time domain sequence x to real and imaginary part of the DFT of x.

	Args:
		x: vector of time domain signal values
		Ux: covariance matrix associated with x, shape (N,N) or noise variance as single float
		N: (optional) length of time domain signal for DFT; N>=Nx
		window: (optional) vector of the time domain window values
		CxCos: cosine part of sensitivity matrix
		CxSin: sine part of sensitivity matrix
		returnC: if true, return sensitivity matrix blocks for later use

	Returns:
		F: vector of complex valued DFT values or of its real and imaginary parts
		UF: covariance matrix associated with real and imaginary part of F
	"""
	L=0
	if isinstance(window,np.ndarray):
		x,Ux = apply_window(x,Ux,window)

	if isinstance(N,int):
		L = N - len(x)
		assert(L>=0)
		x = np.r_[x.copy(),np.zeros(L,)]
	N = len(x)
	if np.mod(N,2) == 0:  # N is even
		M = N+2
	else:
		M = N+1

	F = np.fft.rfft(x)
	F = np.r_[np.real(F),np.imag(F)]

	beta = 2*np.pi*np.arange(N-L)/N

	Cxkc = lambda k: np.cos(k*beta)[np.newaxis,:]
	Cxks = lambda k: -np.sin(k*beta)[np.newaxis,:]

	if isinstance(Ux,float):
		UF = np.zeros(M)
		for k in range(M//2):   # Block cos/cos
				UF[k] = sum(Ux*Cxkc(k)**2)
		for k in range(M//2): # Block sin/sin
				UF[M//2+k] = sum(Ux*Cxks(k)**2)
	else:   # general method
		if len(Ux.shape)==1:
			Ux = np.diag(Ux)
		if not isinstance(CxCos,np.ndarray):
			CxCos = np.zeros((M//2,N-L))
			CxSin = np.zeros((M//2,N-L))
			for k in range(M//2):
				CxCos[k,:] = Cxkc(k)
				CxSin[k,:] = Cxks(k)
		UFCC = np.dot(CxCos,np.dot(Ux,CxCos.T))
		UFCS = np.dot(CxCos,np.dot(Ux,CxSin.T))
		UFSS = np.dot(CxSin,np.dot(Ux,CxSin.T))
		try:
			UF = np.vstack((np.hstack((UFCC,UFCS)),np.hstack((UFCS.T,UFSS))))
		except MemoryError:
			print("Could not put covariance matrix together due to memory constraints.")
			print("Returning the three blocks (A,B,C) such that U = [[A,B],[B.T,C]] instead.")
			UF = (UFCC,UFCS,UFSS)

	if returnC:
		return F,UF,{"CxCos":CxCos,"CxSin":CxSin}
	else:
		return F,UF


def GUM_iDFT(F,UF,Nx=None,Cc=None,Cs=None,returnC=False):
	"""GUM propagation of covariance UF associated with DFT values F to result of
	inverse DFT. The matrix UF is assumed to be for real and imaginary part with blocks:
	UF = [[u(R,R), u(R,I)],[u(I,R),u(I,I)]]
	and real and imaginary part obtained from calling rfft (DFT for real-valued signal)

	Args:
		F: vector of real and imaginary parts of a DFT result
		UF: covariance matrix associated with real and imaginary parts of F
		Nx: number of samples of iDFT result
		Cc: cosine part of sensitivities
		Cs: sine part of sensitivities
		returnC: if true, return sensitivity matrix blocks

	Returns:
		x: vector of time domain signal values
		Ux: covariance matrix associated with x
	"""
	N = UF.shape[0]-2

	if Nx is None:
		Nx = N
	else:
		assert(Nx<=UF.shape[0]-2)

	beta = 2*np.pi*np.arange(Nx)/N

	# calculate inverse DFT
	x = np.fft.irfft(F[:N/2+1]+1j*F[N/2+1:])[:Nx]
	if not isinstance(Cc,np.ndarray):# calculate sensitivities
		Cc = np.zeros((Nx,N//2+1))
		Cc[:,0] = 1.0; Cc[:,-1] = np.cos(np.pi*np.arange(Nx))
		for k in range(1,N//2):
			Cc[:,k] = 2*np.cos(k*beta)

	if not isinstance(Cs,np.ndarray):
		Cs = np.zeros((Nx,N/2+1))
		Cs[:,0] = 0.0; Cs[:,-1] = -np.sin(np.pi*np.arange(Nx))
		for k in range(1,N//2):
			Cs[:,k] = -2*np.sin(k*beta)

	# calculate blocks of uncertainty matrix
	if len(UF.shape)==2:
		RR = UF[:N/2+1,:N/2+1]
		RI = UF[:N/2+1, N/2+1:]
		II = UF[N/2+1:,N/2+1:]
		# propagate uncertainties
		Ux = np.dot(Cc,np.dot(RR,Cc.T))
		Ux = Ux + 2*np.dot(Cc,np.dot(RI,Cs.T))
		Ux = Ux + np.dot(Cs,np.dot(II,Cs.T))
	else:
		RR = UF[:N/2+1]
		II = UF[N/2+1:]
		Ux = np.dot(Cc,prod(RR,Cc.T)) + np.dot(Cs,prod(II,Cs.T))

	if returnC:
		return x,Ux/N**2,{"Cc":Cc,"Cs":Cs}
	else:
		return x,Ux/N**2


def DFT2AmpPhase(F,UF,keep_sparse=False, tol=1.0):
	"""Calculate the matrix U_AP = [[U1,U2],[U2^T,U3]] associated with amplitude and phase of the vector F=[real,imag]
	with associated covariance matrix U_F=[[URR,URI],[URI^T,UII]]

	Args:
		F: vector of real and imaginary parts of a DFT result
		UF: covariance matrix associated with F
		keep_sparse: if true then UAP will be sparse if UF is one-dimensional
		tol: lower bound for A/uF below which a warning will be issued concerning unreliable results
	Returns:
		A: vector of amplitude values
		P: vector of phase values
		UAP: covariance matrix associated with (A,P)
	"""
	# calculate inverse DFT
	N = len(F)-2
	R = F[:N//2+1]; I = F[N//2+1:]

	A  = np.sqrt(R**2+I**2)
	P  = np.arctan2(I,R)
	if len(UF.shape)==1:
		uF = 0.5*(np.sqrt(UF[:N//2+1])+np.sqrt(UF[N//2+1:]))
	else:
		uF = 0.5*(np.sqrt(UF[:N//2+1,:N//2+1])+ np.sqrt(UF[:N//2+1,N//2+1:]))
	if np.any(A/uF < tol):
		print( 'Some amplitude values are below the defined threshold.')
		print('The GUM formulas may become unreliable and a Monte Carlo approach is recommended instead.')
		print('minimum value of A/uF is %.2e and the threshold is %.2e'%((A/uF).min(), tol))
	aR = R/A
	aI = I/A
	pR = -I/A**2
	pI = R/A**2

	if len(UF.shape)==1:
		URR = UF[:N//2+1]
		UII = UF[N//2+1:]
		U11 = URR*aR**2 + UII*aI**2
		U12 = aR*URR*pR + aI*UII*pI
		U22 = URR*pR**2 + UII*pI**2
		UAP = sparse.diags([np.r_[U11,U22],U12,U12],[0,N//2+1,-(N//2+1)])
		if not keep_sparse:
			UAP = UAP.toarray()
	else:
		URR = UF[:N//2+1,:N//2+1]
		URI = UF[:N//2+1,N//2+1:]
		UII = UF[N//2+1:,N//2+1:]
		U11 = prod(aR,prod(URR,aR)) + prod(aR,prod(URI,aI)) + prod(aI,prod(URI.T,aR)) + prod(aI,prod(UII,aI))
		U12 = prod(aR,prod(URR,pR)) + prod(aI,prod(URI,pI)) + prod(aI,prod(URI.T,pR)) + prod(aI,prod(UII,pI))
		U22 = prod(pR,prod(URR,pR)) + prod(pR,prod(URI,pI)) + prod(pI,prod(URI.T,pR)) + prod(pI,prod(UII,pI))
		UAP = np.vstack((np.hstack((U11,U12)),np.hstack((U12.T,U22))))

	return A,P,UAP


def AmpPhase2DFT(A,P,UAP,keep_sparse=False):
	"""Calculate the vector F=[real,imag] and propagate the covariance matrix UAP

	Args:
		A: vector of amplitude values
		P: vector of phase values (in radians)
		UAP: covariance matrix associated with (A,P)
			or vector of squared standard uncertainties [u^2(A),u^2(P)]

	Returns:
		F: vector of real and imaginary parts of DFT result
		UF: covariance matrix associated with F

	"""

	assert(len(A.shape)==1)
	assert(A.shape==P.shape)
	assert(UAP.shape==(2*len(A),2*len(A)) or UAP.shape==(2*len(A),))
	# calculation of F
	F = np.r_[A*np.cos(P),A*np.sin(P)]

	# calculation of sensitivities
	CRA = np.cos(P)
	CRP = -A*np.sin(P)
	CIA = np.sin(P)
	CIP = A*np.cos(P)

	# assignment of uncertainty blocks in UAP
	N = len(A)
	if UAP.shape==(2*N,): # zero correlation; just standard deviations
		Ua = UAP[:N]
		Up = UAP[N:]
		U11 = CRA*Ua*CRA + CRP*Up*CRP
		U12 = CRA*Ua*CIA + CRP*Up*CIP
		U22 = CIA*Ua*CIA + CIP*Up*CIP
		UF = sparse.diags([np.r_[U11,U22],U12,U12],[0,N,-N])
		if not keep_sparse:
			UF = UF.toarray()
	else:
		if isinstance(UAP,sparse.dia_matrix):
			nrows = 2*N
			offset= UAP.offsets
			diags = UAP.data
			Uaa = diags[0][:N]
			Uap = diags[1][offset[1]:nrows+offset[1]]
			Upp = diags[0][N:]

			U11 = Uaa*CRA**2 + CRP*Uap*CRA + CRA*Uap*CRP + Upp*CRP**2
			U12 = CRA*Uaa*CIA + CRP*Uap*CIA + CRA*Uap*CIA + CRP*Upp*CIP
			U22 = Uaa*CIA**2 + CIP*Uap*CIA + CIA*Uap*CIP + Upp*CIP**2

			UF = sparse.diags([np.r_[U11,U22],U12,U12],[0,N,-N])
			if not keep_sparse:
				UF = UF.toarray()
		else:
			Uaa = UAP[:N,:N]
			Uap = UAP[:N,N:]
			Upp = UAP[N:,N:]

			U11 = prod(CRA,prod(Uaa,CRA)) + prod(CRP,prod(Uap.T,CRA)) + prod(CRA,prod(Uap,CRP)) + prod(CRP,prod(Upp,CRP))
			U12 = prod(CRA,prod(Uaa,CIA)) + prod(CRP,prod(Uap.T,CIA)) + prod(CRA,prod(Uap,CIA)) + prod(CRP,prod(Upp,CIP))
			U22 = prod(CIA,prod(Uaa,CIA)) + prod(CIP,prod(Uap.T,CIA)) + prod(CIA,prod(Uap,CIP)) + prod(CIP,prod(Upp,CIP))

			UF = np.vstack((np.hstack((U11,U12)),np.hstack((U12.T,U22))))

	return F, UF

def Time2AmpPhase(x,Ux):
	F,UF = GUM_DFT(x,Ux)
	A,P,UPA = DFT2AmpPhase(F,UF)
	return A,P,UPA


def AmpPhase2Time(A,P,UAP):
	"""GUM propagation of covariance matrix UPA associated with DFT amplitude A and phase P to result of
	inverse DFT. Uncertainty UPA is assumed to be given for amplitude and phase with blocks:
	UPA = [[u(A,A), u(A,P)],[u(P,A),u(P,P)]]

	Sensitivity matrix is designed block-wise as CF = [C_cos,C_sin] such that
	C*UF*C^T = C_cos*u(A,A)*C_cos^T + 2*C_cos*u(A,P)*C_sin^T + C_sin*u(P,P)*C_sin^T

	Args:
		A: vector of amplitude values
		P: vector of phase values (in rad)
		UAP: covariance matrix associated with [A,P]

	Returns:
		x: vector of time domain values
		Ux: covariance matrix associated with x
	"""

	N = UAP.shape[0]-2
	assert(np.mod(N,2)==0)
	beta = 2*np.pi*np.arange(N)/N

	# calculate inverse DFT
	F = A*np.exp(1j*P)
	x = np.irfft(F)

	Pf = np.r_[P,-P[-2:0:-1]]
	Cc = np.zeros((N,N/2+1))
	Cc[:,0] = np.cos(P[0]); Cc[:,-1] = np.cos(P[-1]+np.pi*np.arange(N))
	for k in range(1,N/2):
		Cc[:,k] = 2*np.cos(Pf[k]+k*beta)

	Cs = np.zeros((N,N/2+1))
	Cs[:,0] = -A[0]*np.sin(P[0]); Cs[:,-1] = -A[-1]*np.sin(P[-1]+np.pi*np.arange(N))
	for k in range(1,N/2):
		Cs[:,k] = -A[k]*2*np.sin(Pf[k]+k*beta)

	# calculate blocks of uncertainty matrix
	if len(UAP.shape)==1:
		AA = UAP[:N//2+1]
		PP = UAP[N//2+1:]
		Ux = np.dot(Cc,prod(AA,Cc.T)) + np.dot(Cs,prod(PP,Cs.T))
	else:
		if isinstance(UAP,sparse.dia_matrix):
			nrows = UAP.shape[0]
			n = nrows/2
			offset= UAP.offsets
			diags = UAP.data
			AA = diags[0][:n]
			AP = diags[1][offset[1]:nrows+offset[1]]
			PP = diags[0][n:]
			Ux = np.dot(Cc,prod(AA,Cc.T)) + 2*np.dot(Cc,prod(AP,Cs.T)) + np.dot(Cs,prod(PP,Cs.T))
		else:
			AA = UAP[:N/2+1,:N/2+1]
			AP = UAP[:N/2+1, N/2+1:]
			PP = UAP[N/2+1:,N/2+1:]
			# propagate uncertainties
			Ux = np.dot(Cc,np.dot(AA,Cc.T)) + 2*np.dot(Cc,np.dot(AP,Cs.T)) + np.dot(Cs,np.dot(PP,Cs.T))

	return x,Ux/N**2

# for backward compatibility
GUMdeconv = lambda H, Y, UH, UY: GUM_deconv(H, Y, UH, UY)

def GUM_deconv(H,Y,UH,UY):
	"""GUM propagation of uncertainties for the deconvolution Y = X/H with X and H being the Fourier transform of the measured signal
	and of the system's impulse response, respectively.

	Args:
		H: real and imaginary parts of frequency response values (N an even integer)
		Y: real and imaginary parts of DFT values
		UH: covariance matrix associated with real and imaginary parts of H
		UY: covariance matrix associated with real and imaginary parts of X

	Returns:
		X: real and imaginary parts of DFT values of deconv result
		UX: covariance matrix associated with real and imaginary part of X

	"""
	assert(len(H)==len(Y))
	assert(UH.shape==(len(H),len(H)))
	if len(UY.shape)==2:
		assert(UH.shape==UY.shape)

	N = UH.shape[0]-2
	assert(np.mod(N,2)==0)

# real and imaginary parts of system and signal
	rH, iH = H[:N/2+1], H[N/2+1:]
	rY, iY = Y[:N/2+1], Y[N/2+1:]

	Yc = Y[:N/2+1] + 1j*Y[N/2+1:]
	Hc = H[:N/2+1] + 1j*H[N/2+1:]
	X = np.r_[np.real(Yc/Hc),np.imag(Yc/Hc)]

# sensitivities
	norm = rH**2+iH**2
	RY = np.r_[rH/norm,iH/norm]
	IY = np.r_[-iH/norm,rH/norm]
	RH = np.r_[(-rY*rH**2+rY*iH**2-2*iY*iH*rH)/norm**2, (iY*rH**2-iH*iH**2-2*rY*rH*iH)/norm**2]
	IH = np.r_[(-iY*rH**2+iY*iH**2+2*rY*iH*rH)/norm**2, (-rY*rH**2+rY*iH**2-2*iY*rH*iH)/norm**2]
# calculate blocks of uncertainty matrix
	URRX = matprod(UY,RY,RY) + matprod(UH,RH,RH)
	URIX = matprod(UY,RY,IY) + matprod(UH,RH,IH)
	UIIX = matprod(UY,IY,IY) + matprod(UH,IH,IH)

	try:
		UX = np.vstack((np.hstack((URRX,URIX)),np.hstack((URIX.T,UIIX))))
	except MemoryError:
		print("Could not put covariance matrix together due to memory constraints.")
		print("Returning the three blocks (A,B,C) such that U = [[A,B],[B.T,C]] instead.")
		UX = (URRX,URIX,UIIX)

	return X,UX


def GUM_multiply(Y, UY, F, UF=None):
	"""
	GUM uncertainty propagation for multiplication in the frequency domain, where the second factor F may have an
	associated uncertainty.

	Args:
	    Y: real and imaginary parts of the first factor
	    UY: covariance matrix or squared uncertainty associated with Y
	    F: real and imaginary parts of the second factor
	    UF: covariance matrix associated with F (optional), default is None

	Returns:
		The product YF and the associated uncertainty.
	"""

	assert(len(Y)==len(F))

	def calcU(A, UB):
		# uncertainty propagation for A*B with B uncertain
		n = len(A)
		RA = A[:n//2]
		IA = A[n//2:]
		if isinstance(UB, float):
			uRR = RA * UB * RA + IA * UB * IA
			uRI = RA * UB * IA - IA * UB * RA
			uII = IA * UB * IA + RA * UB * RA
		elif len(UB.shape)==1:
			UBRR = UB[:n//2]
			UBII = UB[n//2:]
			uRR = RA*UBRR*RA + IA*UBII*IA
			uRI = RA*UBRR*IA - IA*UBII*RA
			uII = IA*UBRR*IA + RA*UBII*RA
		else:
			UBRR = UB[:n//2,:n//2]
			UBRI = UB[:n//2,n//2:]
			UBII = UB[n//2:,n//2:]
			uRR = prod(RA, prod(UBRR, RA)) - prod(IA,prod(UBRI.T, RA)) - prod(RA, prod(UBRI,IA)) + prod(IA,prod(UBII,IA))
			uRI = prod(RA, prod(UBRR, IA)) - prod(IA,prod(UBRI.T, IA)) + prod(RA, prod(UBRI,RA)) - prod(IA,prod(UBII,RA))
			uII = prod(IA, prod(UBRR, IA)) + prod(RA,prod(UBRI.T, IA)) + prod(IA, prod(UBRI,RA)) + prod(RA,prod(UBII,RA))
		return uRR, uRI, uII

	N = len(Y)
	RY = Y[:N//2]; IY = Y[N//2:]
	RF = F[:N//2]; IF = F[N//2:]
	YF = np.r_[RY*RF - IY*IF, RY*IF + IY*RF]
	if not isinstance(UF, np.ndarray):  # second factor is known exactly
		UYRR, UYRI, UYII = calcU(F, UY)
		UYF  = np.vstack((np.hstack((UYRR, UYRI)), np.hstack((UYRI.T, UYII))))
	else:
		URR_Y, URI_Y, UII_Y = calcU(F, UY)
		URR_F, URI_F, UII_F = calcU(Y, UF)
		URR = URR_Y + URR_F
		URI = URI_Y + URI_F
		UII = UII_Y + UII_F
		UYF = np.vstack((np.hstack((URR, URI)), np.hstack((URI.T, UII))))
	return YF, UYF




