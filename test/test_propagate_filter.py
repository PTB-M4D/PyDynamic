"""

Uncertainty propagation for a FIR lowpass filter with uncertain cut-off frequency for a rectangular signal.

"""

import matplotlib.pyplot as plt
import numpy as np

from PyDynamic.misc.testsignals import rect
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter
from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.misc.filterstuff import kaiser_lowpass
import PyDynamic.misc.noise as pmn
import PyDynamic.uncertainty.propagate_MonteCarlo as MC

# import PyDynamic from local code, not from the (possibly installed) module
import sys
sys.path.append(".")


def test_FIR(makePlots=False, corrNoise=False):

    # parameters of simulated measurement
    Fs = 100e3        # sampling frequency (in Hz)
    Ts = 1 / Fs       # sampling interval length (in s)

    # nominal system parameters
    fcut = 20e3                            # low-pass filter cut-off frequency (6 dB)
    L = 100                                # filter order
    b = kaiser_lowpass(L,fcut,Fs)[0]

    # uncertain knowledge: cutoff between 19.5kHz and 20.5kHz
    runs = 1000
    FC = fcut + (2*np.random.rand(runs)-1)*0.5e3

    B = np.zeros((runs,L+1))
    for k in range(runs):        # Monte Carlo for filter coefficients of low-pass filter
        B[k,:] = kaiser_lowpass(L,FC[k],Fs)[0]

    Ub = make_semiposdef(np.cov(B,rowvar=0))    # covariance matrix of MC result

    # simulate input and output signals
    nTime = 500
    time  = np.arange(nTime)*Ts                     # time values
    sigma_noise = 1e-2                              # 1e-5

    # use white or non-white (correlated) noise
    if corrNoise:
        color = "white"
        w      = np.random.normal(loc = 0, scale = sigma_noise, size=nTime)
        cn     = corr_noise(w, sigma_noise)
        Ux     = cn.theoretic_covariance_colored_noise(N=nTime, color=color)
        noise  = cn.colored_noise(color=color)

    else:
        noise  = sigma_noise

    # input signal
    x = rect(time,100*Ts,250*Ts,1.0,noise=noise)

    # run methods
    y, Uy = FIRuncFilter(x, noise, b, Ub, blow=b)            # apply uncertain FIR filter (GUM formula)
    yMC,UyMC = MC.MC(x,sigma_noise,b,[1.0],Ub,runs=1000,blow=b)    # apply uncertain FIR filter (Monte Carlo)

    # plot if necessary
    if makePlots:
        plt.figure(1); plt.cla()
        plt.plot(time, x, label="input")
        plt.plot(time, y, label="output")
        plt.xlabel("time / au")
        plt.ylabel("signal amplitude / au")
        plt.legend()

        plt.figure(2);plt.cla()
        plt.plot(time, Uy, label="FIR formula")
        plt.plot(time, np.sqrt(np.diag(UyMC)), label="Monte Carlo")
        plt.xlabel("time / au")
        plt.ylabel("signal uncertainty/ au")
        plt.legend()
        plt.show()
