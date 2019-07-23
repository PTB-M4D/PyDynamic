"""

Uncertainty propagation for a FIR lowpass filter with uncertain cut-off frequency for a rectangular signal.

"""

import matplotlib.pyplot as plt
import numpy as np

from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.misc.filterstuff import kaiser_lowpass
import PyDynamic.misc.noise as pmn
from PyDynamic.misc.testsignals import rect
import PyDynamic.uncertainty.propagate_MonteCarlo as MC
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter


def test_sigma_noise(makePlots=False):

    # parameters of simulated measurement
    Fs = 100e3        # sampling frequency (in Hz)
    Ts = 1 / Fs       # sampling interval length (in s)

    # nominal system parameters
    fcut = 20e3                            # low-pass filter cut-off frequency (6 dB)
    L = 100                                 # filter order
    b1 = kaiser_lowpass(L,   fcut,Fs)[0]
    b2 = kaiser_lowpass(L-20,fcut,Fs)[0]

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

    # different cases
    sigma_noise = 1e-2                              # 1e-5

    for kind in ["float", "corr", "diag"]:

        print(kind)

        if kind == "float":
            # input signal + run methods
            x = rect(time,100*Ts,250*Ts,1.0,noise=sigma_noise)
            y, Uy = FIRuncFilter(x, sigma_noise, b1, Ub, blow=b2, kind=kind)            # apply uncertain FIR filter (GUM formula)
            
            assert len(y) == len(x)
            assert len(Uy) == len(x)

        elif kind == "corr":
            color = "white"
            Ux     = pmn.power_law_acf(nTime, color=color, std=sigma_noise)
            noise  = pmn.power_law_noise(N=nTime, color=color, std=sigma_noise)

            # input signal + run methods
            x = rect(time,100*Ts,250*Ts,1.0,noise=noise)
            y, Uy = FIRuncFilter(x, Ux, b1, Ub, blow=b2, kind=kind)            # apply uncertain FIR filter (GUM formula)

            assert len(y) == len(x)
            assert len(Uy) == len(x)

        
        elif kind == "diag":
            pass

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


test_sigma_noise()