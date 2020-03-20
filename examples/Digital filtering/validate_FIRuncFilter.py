"""
Perform validation for uncertainty.propagate_filter with (MC) methods
"""
import sys
sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz

from PyDynamic.misc.testsignals import rect
from PyDynamic.misc.tools import make_semiposdef, trimOrPad
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

for kind in ["float", "corr", "diag"]:

    for blow in [None, b2]:

        print(kind, type(blow))

        if kind == "float":
            # input signal + run methods
            x = rect(time,100*Ts,250*Ts,1.0,noise=sigma_noise)

            y, Uy = FIRuncFilter(x, sigma_noise, b1, Ub, blow=blow, kind=kind)       # apply uncertain FIR filter (GUM formula)
            yMC, UyMC = MC(x,sigma_noise,b1,[1.0],Ub,runs=runs,blow=blow)             # apply uncertain FIR filter (Monte Carlo)
            
        elif kind == "corr":

            # get an instance of noise, the covariance and the covariance-matrix with the specified color
            color = "red"
            noise = power_law_noise(N=nTime, color_value=color, std=sigma_noise)

            Ux = power_law_acf(nTime, color_value=color, std=sigma_noise)

            # input signal
            x = rect(time,100*Ts,250*Ts,1.0,noise=noise)

            # build Ux_matrix from autocorrelation Ux
            Ux_matrix = toeplitz(trimOrPad(Ux, nTime))

            # run methods
            y, Uy = FIRuncFilter(x, Ux, b1, Ub, blow=blow, kind=kind)              # apply uncertain FIR filter (GUM formula)
            yMC, UyMC = MC(x, Ux_matrix, b1, [1.0], Ub, runs=runs, blow=blow)             # apply uncertain FIR filter (Monte Carlo)

        elif kind == "diag":
            sigma_diag = sigma_noise * ( 1 + np.heaviside(np.arange(len(time)) - len(time)//2.5,0) )    # std doubles after half of the time
            noise = sigma_diag * white_gaussian(len(time))

            # input signal + run methods
            x = rect(time,100*Ts,250*Ts,1.0,noise=noise)

            y, Uy = FIRuncFilter(x, sigma_diag, b1, Ub, blow=blow, kind=kind)            # apply uncertain FIR filter (GUM formula)
            yMC, UyMC = MC(x, sigma_diag, b1, [1.0], Ub, runs=runs, blow=blow)             # apply uncertain FIR filter (Monte Carlo)

        # compare FIR and MC results
        plt.figure(1); plt.cla()
        plt.plot(time, x, label="input")
        plt.plot(time, y, label="output FIR direct")
        plt.plot(time, yMC, label="output FIR MC")
        plt.xlabel("time [s]")
        plt.ylabel("signal amplitude [1]")
        plt.legend()

        plt.figure(2);plt.cla()
        plt.plot(time, Uy, label="FIR formula")
        plt.plot(time, np.sqrt(np.diag(UyMC)), label="Monte Carlo")
        plt.xlabel("time [s]")
        plt.ylabel("signal uncertainty [1]")
        plt.legend()
        plt.show()