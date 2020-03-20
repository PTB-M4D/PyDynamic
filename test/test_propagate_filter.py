"""
Perform test for uncertainty.propagate_filter
"""

import matplotlib.pyplot as plt
import numpy as np

from PyDynamic.misc.testsignals import rect
from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.misc.filterstuff import kaiser_lowpass
from PyDynamic.misc.noise import power_law_acf, power_law_noise, white_gaussian
from PyDynamic.uncertainty.propagate_MonteCarlo import MC
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter

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


def test_FIRuncFilter_float():

    # input signal + run methods
    x = rect(time,100*Ts,250*Ts,1.0,noise=sigma_noise)

    # apply uncertain FIR filter (GUM formula)
    for blow in [None, b2]:
        y, Uy = FIRuncFilter(x, sigma_noise, b1, Ub, blow=blow, kind="float")
        assert len(y) == len(x)
        assert len(Uy) == len(x)

def test_FIRuncFilter_corr():

    # get an instance of noise, the covariance and the covariance-matrix with the specified color
    color = "white"
    noise = power_law_noise(N=nTime, color_value=color, std=sigma_noise)
    Ux = power_law_acf(nTime, color_value=color, std=sigma_noise)

    # input signal
    x = rect(time,100*Ts,250*Ts,1.0,noise=noise)

    # apply uncertain FIR filter (GUM formula)
    for blow in [None, b2]:
        y, Uy = FIRuncFilter(x, Ux, b1, Ub, blow=blow, kind="corr")
        assert len(y) == len(x)
        assert len(Uy) == len(x)
    
def test_FIRuncFilter_diag():
    sigma_diag = sigma_noise * ( 1 + np.heaviside(np.arange(len(time)) - len(time)//2,0) )    # std doubles after half of the time
    noise = sigma_diag * white_gaussian(len(time))

    # input signal + run methods
    x = rect(time,100*Ts,250*Ts,1.0,noise=noise)

    # apply uncertain FIR filter (GUM formula)
    for blow in [None, b2]:
        y, Uy = FIRuncFilter(x, sigma_diag, b1, Ub, blow=blow, kind="diag")
        assert len(y) == len(x)
        assert len(Uy) == len(x)

def test_IIRuncFilter():
    pass