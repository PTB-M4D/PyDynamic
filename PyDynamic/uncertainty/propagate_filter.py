# -*- coding: utf-8 -*-




import numpy as np
from scipy.signal import lfilter,tf2ss
from scipy.linalg import toeplitz
from ..misc.tools import zerom

# TODO Implement formula for colored noise
# TODO Implement formula for covariance calculation
# TODO Allow zero uncertainty for filter
def FIRuncFilter(y,sigma_noise,theta,Utheta=None,shift=0,blow=None):
    """Uncertainty propagation for signal y and uncertain FIR filter theta

    Parameters
    ----------
        y: np.ndarray
            filter input signal
        sigma_noise: np.ndarray
            standard deviation of white noise in y
        theta: np.ndarray
            FIR filter coefficients
        Utheta: np.ndarray
            squared uncertainty associated with theta
        shift: int
            time delay of filter output signal (in samples)
        blow: np.ndarray
            optional FIR low-pass filter

    Returns
    -------
        x: np.ndarray
            FIR filter output signal
        ux: np.ndarray
            point-wise uncertainties associated with x


    References
    ----------
        * Elster and Link 2008 [Elster2008]_

    .. seealso:: :mod:`PyDynamic.deconvolution.fit_filter`

    Todo:
        * Implement formula for colored noise
        * Implement formula for covariance calculation

    """
    if not isinstance(Utheta, np.ndarray):
        Utheta = np.zeros((len(theta), len(theta)))

    if isinstance(blow,np.ndarray):
        LR = 600
        Bcorr = np.correlate(blow,blow,mode='full')
        if isinstance(sigma_noise,float):
            ycorr = np.convolve(sigma_noise**2,Bcorr)
        else:
            if len(sigma_noise.shape)==1:
                assert (len(sigma_noise)==len(y)), "Length of uncertainty and signal are inconsistent"
                ycorr = np.convolve(sigma_noise, Bcorr)
            else:
                raise NotImplementedError("FIR formula for covariance propagation not implemented. Suggest Monte Carlo propagation instead.")
        Lr = len(ycorr)
        Lstart = int(np.ceil(Lr/2.0))
        Lend = Lstart + LR -1
        Ryy = toeplitz(ycorr[Lstart:Lend])
        Ulow= Ryy[:len(theta),:len(theta)]
        xlow = lfilter(blow,1.0,y)
    else:
        if isinstance(sigma_noise, float):
            Ulow = np.eye(len(theta))*sigma_noise**2
        else:
            if len(sigma_noise.shape)==1:
                Ulow = np.diag(sigma_noise)
            else:
                Ulow = sigma_noise
        xlow = y


    x = lfilter(theta,1.0,xlow)
    x = np.roll(x,int(shift))

    UncCov = np.dot(theta[:,np.newaxis].T,np.dot(Ulow,theta)) + np.abs(np.trace(np.dot(Ulow,Utheta)))

    L = len(theta)

    unc = np.zeros_like(y)
    unc[:L] = 0.0
    for m in range(L,len(y)):
        XL = xlow[m:m-L:-1]
        unc[m] = np.dot(XL[:,np.newaxis].T,np.dot(Utheta,XL[:,np.newaxis]))

    ux = np.sqrt(np.abs(UncCov + unc))
    ux = np.roll(ux,int(shift))

    return x, ux


#TODO: Remove utilization of numpy.matrix
#TODO: Extend to colored noise
#TODO: Allow zero uncertainty for filter
def IIRuncFilter(x, noise, b, a, Uab):
    """Uncertainty propagation for the signal x and the uncertain IIR filter (b,a)

    Parameters
    ----------
	    x: np.ndarray
	        filter input signal
	    noise: float
	        signal noise standard deviation
	    b: np.ndarray
	        filter numerator coefficients
	    a: np.ndarray
	        filter denominator coefficients
	    Uab: np.ndarray
	        covariance matrix for (a[1:],b)

    Returns
    -------
	    y: np.ndarray
	        filter output signal
	    Uy: np.ndarray
	        uncertainty associated with y

    References
    ----------
        * Link and Elster [Link2009]_

    .. seealso:: :mod:`PyDynamic.uncertainty.propagate_MonteCarlo.SMC`

	"""

    if not isinstance(noise, np.ndarray):
        noise = noise * np.ones(np.shape(x))

    p = len(a) - 1
    if not len(b) == len(a):
        b = np.hstack((b, np.zeros((len(a) - len(b),))))

    # from discrete-time transfer function to state space representation
    [A, bs, c, b0] = tf2ss(b, a)

    A = np.matrix(A)
    bs = np.matrix(bs)
    c = np.matrix(c)

    phi = zerom((2 * p + 1, 1))
    dz = zerom((p, p))
    dz1 = zerom((p, p))
    z = zerom((p, 1))
    P = zerom((p, p))

    y = np.zeros((len(x),))
    Uy = np.zeros((len(x),))

    Aabl = np.zeros((p, p, p))
    for k in range(p):
        Aabl[0, k, k] = -1

    for n in range(len(y)):
        for k in range(p):  # derivative w.r.t. a_1,...,a_p
            dz1[:, k] = A * dz[:, k] + np.squeeze(Aabl[:, :, k]) * z
            phi[k] = c * dz[:, k] - b0 * z[k]
        phi[p + 1] = -np.matrix(a[1:]) * z + x[n]  # derivative w.r.t. b_0
        for k in range(p + 2, 2 * p + 1):  # derivative w.r.t. b_1,...,b_p
            phi[k] = z[k - (p + 1)]
        P = A * P * A.T + noise[n] ** 2 * (bs * bs.T)
        y[n] = c * z + b0 * x[n]
        Uy[n] = phi.T * Uab * phi + c * P * c.T + b[0] ** 2 * noise[n] ** 2
        # update of the state equations
        z = A * z + bs * x[n]
        dz = dz1

    Uy = np.sqrt(np.abs(Uy))

    return y, Uy


