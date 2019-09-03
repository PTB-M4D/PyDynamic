# -*- coding: utf-8 -*-
""" Perform tests on the method *uncertainty.propagate_MonteCarlo*"""

import numpy as np
from pytest import raises

from PyDynamic.misc.testsignals import rect
from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.misc.filterstuff import kaiser_lowpass
#from PyDynamic.misc.noise import power_law_acf, power_law_noise, white_gaussian, ARMA
from PyDynamic.uncertainty.propagate_MonteCarlo import MC, SMC, UMC, ARMA

import matplotlib.pyplot as plt

def test_MC():
    # maybe take this test from some example?
    pass


def test_SMC():
    # maybe take this test from some example?
    pass


def test_UMC(visualizeOutput=False):
    # parameters of simulated measurement
    Fs = 100e3        # sampling frequency (in Hz)
    Ts = 1 / Fs       # sampling interval length (in s)

    # nominal system parameters
    fcut = 20e3                                 # low-pass filter cut-off frequency (6 dB)
    L = 100                                     # filter order
    b1 = kaiser_lowpass(L,   fcut,Fs)[0]
    b2 = kaiser_lowpass(L-20,fcut,Fs)[0]

    # uncertain knowledge: cutoff between 19.5kHz and 20.5kHz
    runs = 1000
    FC = fcut + (2*np.random.rand(runs)-1)*0.5e3

    B = np.zeros((runs,L+1))
    for k in range(runs):                       # Monte Carlo for filter coefficients of low-pass filter
        B[k,:] = kaiser_lowpass(L,FC[k],Fs)[0]

    Ub = make_semiposdef(np.cov(B,rowvar=0))    # covariance matrix of MC result

    # simulate input and output signals
    nTime = 500
    time  = np.arange(nTime)*Ts                 # time values

    # different cases
    sigma_noise = 1e-5

    for kind in ["float", "corr", "diag"]:

        print(kind)

        if kind == "float":
            # input signal + run methods
            x = rect(time,100*Ts,250*Ts,1.0,noise=sigma_noise)

            # run method
            #yMC,UyMC = MC(x,sigma_noise,b1,[1.0],Ub,runs=runs,blow=b2)             # apply uncertain FIR filter (Monte Carlo)

            #yUMC, UyUMC = UMC(x, b1, [1.0], Ub, sigma=sigma_noise, runs=200, Delta=0.001)
            yUMC, UyUMC, p025, p975, happr = UMC(x, b1, [1.0], Ub, sigma=sigma_noise, runs=20, runs_init=10, nbins=10, verbose_return=True)
            
            assert len(yUMC) == len(x)
            assert len(UyUMC) == len(x)
            assert p025.shape[1] == len(x)
            assert p975.shape[1] == len(x)
            assert isinstance(happr, dict)

            if visualizeOutput:
                # visualize input and mean of system response
                plt.plot(time, x)
                plt.plot(time, yUMC)

                # visualize uncertainty of output
                plt.plot(time, yUMC - UyUMC, linestyle="--", linewidth=1, color="red")
                plt.plot(time, yUMC + UyUMC, linestyle="--", linewidth=1, color="red")

                # visualize central 95%-quantile
                plt.plot(time, p025.T, linestyle=":", linewidth=1, color="gray")
                plt.plot(time, p975.T, linestyle=":", linewidth=1, color="gray")

                # visualize the bin-counts
                key = list(happr.keys())[0]
                for ts, be, bc in zip(time, happr[key]["bin-edges"].T, happr[key]["bin-counts"].T):
                    plt.scatter(ts*np.ones_like(bc), be[1:], bc)
                    
                plt.show()
            
        #elif kind == "corr":

        #    # get an instance of noise, the covariance and the covariance-matrix with the specified color
        #    color = "white"
        #    noise = power_law_noise(N=nTime, color=color, std=sigma_noise)
        #    Ux_matrix = power_law_acf(nTime, color=color, std=sigma_noise, returnMatrix=True)

        #    # input signal
        #    x = rect(time,100*Ts,250*Ts,1.0,noise=noise)

        #    # run method
        #    yMC,UyMC = MC(x,Ux_matrix,b1,[1.0],Ub,runs=runs,blow=b2)             # apply uncertain FIR filter (Monte Carlo)

        #    assert len(y) == len(x)
        #    assert len(Uy) == len(x)
        
        #elif kind == "diag":
        #    sigma_diag = sigma_noise * ( 1 + np.heaviside(np.arange(len(time)) - len(time)//2,0) )    # std doubles after half of the time
        #    noise = sigma_diag * white_gaussian(len(time))

        #    # input signal + run methods
        #    x = rect(time,100*Ts,250*Ts,1.0,noise=noise)

        #    # run method
        #    yMC,UyMC = MC(x,sigma_diag,b1,[1.0],Ub,runs=runs,blow=b2)             # apply uncertain FIR filter (Monte Carlo)

def test_noise_ARMA():
    length = 100
    phi = [1/3, 1/4, 1/5]
    theta = [1, -1 ]

    e = ARMA(length, phi = phi, theta = theta)

    assert len(e) == length

test_UMC(visualizeOutput=True)