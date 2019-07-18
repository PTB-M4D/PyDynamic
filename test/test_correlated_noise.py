import numpy as np

### TESTING
import sys
sys.path.append(".")
### /TESTING

from PyDynamic.misc.tools import col_hstack, make_semiposdef
from PyDynamic.misc.filterstuff import kaiser_lowpass
from PyDynamic.misc.testsignals import corr_noise
from PyDynamic.misc.testsignals import rect
import PyDynamic.uncertainty.propagate_MonteCarlo as MC
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter


def test_sigma_noise():

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

        elif kind == "corr":
            color = "white"
            w      = np.random.normal(loc = 0, scale = sigma_noise, size=nTime)
            cn     = corr_noise(w, sigma_noise)
            Ux     = cn.theoretic_covariance_colored_noise(N=nTime, color=color)
            noise  = cn.colored_noise(color=color)

            # input signal + run methods
            x = rect(time,100*Ts,250*Ts,1.0,noise=noise)
            y, Uy = FIRuncFilter(x, Ux, b1, Ub, blow=b2, kind=kind)            # apply uncertain FIR filter (GUM formula)

        
        elif kind == "diag":
            pass



test_sigma_noise()