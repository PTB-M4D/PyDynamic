# -*- coding: utf-8 -*-

"""

Application of the formula for the calculation of variance for the case of FIR filtering

"""

import numpy as np
from scipy.signal import lfilter
from scipy.linalg import toeplitz

# TODO Implement formula for non-trivial noise
# TODO Implement formula for covariance calculation
def FIRuncFilter(y,sigma_noise,theta,Utheta,shift=0,blow=1.0):
    """
    Uncertainty propagation for signal y and uncertain FIR filter theta

    :param y: filter input signal
    :param sigma_noise: standard deviation of white noise in y
    :param theta: FIR filter coefficients
    :param Utheta: uncertainty associated with theta
    :param shift: shift of filter output signal (in samples)
    :param blow: optional FIR low-pass filter

    :returns x: FIR filter output
    :returns ux: associated point-wise uncertainties

    Application of the FIR propagation formula from

    C. Elster and A. Link. (2008)
    Uncertainty evaluation for dynamic measurements modelled by a linear time-invariant system
    Metrologia 45 464-473

    .. seealso:: :mod:`PyDynamic.deconvolution.fit_filter`

    """


    if isinstance(blow,np.ndarray):
        LR = 600
        Bcorr = np.correlate(blow,blow,mode='full')
        ycorr = np.convolve(sigma_noise**2,Bcorr)
        Lr = len(ycorr)
        Lstart = np.ceil(Lr/2.0)
        Lend = Lstart + LR -1
        Ryy = toeplitz(ycorr[Lstart:Lend])
        Ulow= Ryy[:len(theta),:len(theta)]
        xlow = lfilter(blow,1.0,y)
    else:
        Ulow = blow*np.eye(len(theta))*sigma_noise**2
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