# -*- coding: utf-8 -*-
"""
    Fit of a FIR filter to a simulated reciprocal frequency response of a
    second-order dynamic system. The deconvolution filter is applied to the
    simulated response of the system to a shock-like Gaussian signal.
    In this example uncertainties associated with the simulated system are
    propagated to the estimate of the input signal.

"""
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

# if run as script, add parent path for relative importing
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import deconvolution.fit_filter as deconv
import misc.SecondOrderSystem as sos
from misc.testsignals import shocklikeGaussian
from misc.filterstuff import kaiser_lowpass
from uncertainty.propagate_FIR import FIRuncFilter
from misc.tools import make_semiposdef

rst = np.random.RandomState(10)

##### FIR filter parameters
N = 12  # filter order
tau = 6 # time delay

# parameters of simulated measurement
Fs = 500e3
Ts = 1 / Fs

# sensor/measurement system
f0 = 36e3; uf0 = 0.04e3
S0 = 0.124; uS0= 1.3e-5
delta = 0.0055; udelta = 1e-3

# transform continuous system to digital filter
bc, ac = sos.phys2filter(S0,delta,f0)
b, a = dsp.bilinear(bc, ac, Fs)

# simulate input and output signals
time = np.arange(0, 4e-3 - Ts, Ts)
x = shocklikeGaussian(time, t0 = 2e-3, sigma = 1e-5, m0=0.8)
y = dsp.lfilter(b, a, x)
noise = 1e-3
yn = y + np.random.randn(np.size(y)) * noise

# Monte Carlo for calculation of unc. assoc. with [real(H),imag(H)]
runs = 10000
MCS0 = S0 + rst.randn(runs)*uS0
MCd  = delta+ rst.randn(runs)*udelta
MCf0 = f0 + rst.randn(runs)*uf0
f = np.linspace(0, 120e3, 200)
HMC = sos.FreqResp(MCS0, MCd, MCf0, f)

H = np.mean(HMC,dtype=complex,axis=1)
UH= np.cov(np.vstack((np.real(HMC),np.imag(HMC))),rowvar=1)
UH= make_semiposdef(UH)

# Calculation of FIR deconvolution filter and its assoc. unc.
bF, UbF = deconv.LSFIR_unc(H,UH,N,tau,f,Fs)

# correlation of filter coefficients
CbF = UbF/(np.tile(np.sqrt(np.diag(UbF))[:,np.newaxis],(1,N+1))*
		   np.tile(np.sqrt(np.diag(UbF))[:,np.newaxis].T,(N+1,1)))

# Deconvolution Step1: lowpass filter for noise attenuation
fcut = 35e3; low_order = 100
blow, lshift = kaiser_lowpass(low_order, fcut, Fs)
shift = -tau - lshift
# Deconvolution Step2: Deconvolution filter
xhat,Uxhat = FIRuncFilter(yn,noise,bF,UbF,shift,blow)


# Plot of results
# fplot = np.linspace(0, 80e3, 1000)
# Hc = sos.FreqResp(S0, delta, f0, fplot)
# Hif = dsp.freqz(bF, 1.0, 2 * np.pi * fplot / Fs)[1]
#
# plt.figure(1); plt.clf()
# plt.plot(fplot, db(Hc), fplot, db(Hif), fplot, db(Hc*Hif))
# plt.legend(('System', 'FIR deconv fit','compensation result'))
# plt.title('Amplitude of frequency responses')
#
# fig = plt.figure(2);plt.clf()
# ax = fig.add_subplot(1,1,1)
# plt.imshow(UbF,interpolation="none")
# plt.colorbar(ax=ax)
# plt.title('Uncertainty of deconvolution filter coefficients')
#
# fig = plt.figure(21);plt.clf()
# ax = fig.add_subplot(1,1,1)
# plt.imshow(CbF,interpolation="none")
# plt.colorbar(ax=ax)
#
#
# plt.figure(3); plt.clf()
# plt.plot(time,col_hstack([x,yn,xhat]))
# plt.legend(('input','output','estimate'))
# plt.title('time domain signals')
#
# plt.figure(4);plt.clf()
# plt.plot(time,Uxhat)
# plt.title('Uncertainty of estimated input signal')
#
# plt.show()
