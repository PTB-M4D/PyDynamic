# -*- coding: utf-8 -*-
""" Perform tests on the method *uncertainty.propagate_MonteCarlo*"""

import numpy as np
from pytest import raises
import functools
import scipy

from PyDynamic.misc.testsignals import rect
from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.misc.filterstuff import kaiser_lowpass
#from PyDynamic.misc.noise import power_law_acf, power_law_noise, white_gaussian, ARMA
from PyDynamic.uncertainty.propagate_MonteCarlo import MC, SMC, UMC, ARMA, UMC_generic, _UMCevaluate

import matplotlib.pyplot as plt




##### some definitions for all tests

# parameters of simulated measurement
Fs = 100e3        # sampling frequency (in Hz)
Ts = 1 / Fs       # sampling interval length (in s)

# nominal system parameters
fcut = 20e3                                 # low-pass filter cut-off frequency (6 dB)
L = 100                                     # filter order
b1 = kaiser_lowpass(L,   fcut,Fs)[0]
b2 = kaiser_lowpass(L-20,fcut,Fs)[0]

# uncertain knowledge: cutoff between 19.5kHz and 20.5kHz
runs = 20
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

# input signal + run methods
x = rect(time,100*Ts,250*Ts,1.0,noise=sigma_noise)



##### actual tests

def test_MC(visualizeOutput=False):
    # run method
    y,Uy = MC(x,sigma_noise,b1,[1.0],Ub,runs=runs,blow=b2)

    assert len(y) == len(x)
    assert Uy.shape == (x.size, x.size)

    if visualizeOutput:
        # visualize input and mean of system response
        plt.plot(time, x)
        plt.plot(time, y)

        # visualize uncertainty of output
        plt.plot(time, y - np.sqrt(np.diag(Uy)), linestyle="--", linewidth=1, color="red")
        plt.plot(time, y + np.sqrt(np.diag(Uy)), linestyle="--", linewidth=1, color="red")

        plt.show()


# this does not run through yet
#def test_SMC():
#    # run method
#    y,Uy = SMC(x, sigma_noise, b1, [1.0], Ub, runs=runs)
#
#    assert len(y) == len(x)
#    assert Uy.shape == (x.size, x.size)


def test_UMC(visualizeOutput=False):
    # run method
    y, Uy, p025, p975, happr = UMC(x, b1, [1.0], Ub, blow=b2, sigma=sigma_noise, runs=runs, runs_init=10, nbins=10, verbose_return=True)

    assert len(y) == len(x)
    assert Uy.shape == (x.size, x.size)
    assert p025.shape[1] == len(x)
    assert p975.shape[1] == len(x)
    assert isinstance(happr, dict)

    if visualizeOutput:
        # visualize input and mean of system response
        plt.plot(time, x)
        plt.plot(time, y)

        # visualize uncertainty of output
        plt.plot(time, y - np.sqrt(np.diag(Uy)), linestyle="--", linewidth=1, color="red")
        plt.plot(time, y + np.sqrt(np.diag(Uy)), linestyle="--", linewidth=1, color="red")

        # visualize central 95%-quantile
        plt.plot(time, p025.T, linestyle=":", linewidth=1, color="gray")
        plt.plot(time, p975.T, linestyle=":", linewidth=1, color="gray")

        # visualize the bin-counts
        key = list(happr.keys())[0]
        for ts, be, bc in zip(time, happr[key]["bin-edges"].T, happr[key]["bin-counts"].T):
            plt.scatter(ts*np.ones_like(bc), be[1:], bc)

        plt.show()


def test_UMC_generic(visualizeOutput=False):

    Ux = 1e-2 * np.diag(np.abs(np.sin(time * Fs / 10)))

    drawSamples = lambda size: np.random.multivariate_normal(x, Ux, size)
    #params = {"b": b2, "a":[1]}  # does not work
    evaluate = functools.partial(scipy.signal.lfilter, b2, [1])

    # run UMC
    y, Uy, happr = UMC_generic(drawSamples, evaluate, runs=100, blocksize=20, runs_init=10)

    assert y.size == Uy.shape[0]
    assert Uy.shape == (y.size, y.size)
    assert isinstance(happr, dict)

    # run again, but only return all simulations
    sims = UMC_generic(drawSamples, evaluate, runs=100, blocksize=20, runs_init=10, return_simulations=True)
    assert isinstance(sims, dict)

    if visualizeOutput:
        # visualize input and mean of system response
        plt.plot(time, x)
        plt.plot(time, y)

        # visualize uncertainty of output
        plt.plot(time, y - np.sqrt(np.diag(Uy)), linestyle="--", linewidth=1, color="red")
        plt.plot(time, y + np.sqrt(np.diag(Uy)), linestyle="--", linewidth=1, color="red")

        # visualize the bin-counts
        key = list(happr.keys())[0]
        for ts, be, bc in zip(time, happr[key]["bin-edges"].T, happr[key]["bin-counts"].T):
            plt.scatter(ts*np.ones_like(bc), be[1:], bc)

        plt.show()


def test_compare_MC_UMC():

    y_MC, Uy_MC = MC(x,sigma_noise,b1,[1.0],Ub,runs=2*runs,blow=b2)
    y_UMC, Uy_UMC = UMC(x, b1, [1.0], Ub, blow=b2, sigma=sigma_noise, runs=2*runs, runs_init=10)

    # both methods should yield roughly the same results
    assert np.allclose(y_MC, y_UMC, atol=1e-3)
    assert np.allclose(Uy_MC, Uy_UMC, atol=1e-3)


def test_noise_ARMA():
    length = 100
    phi = [1/3, 1/4, 1/5]
    theta = [1, -1 ]

    e = ARMA(length, phi = phi, theta = theta)

    assert len(e) == length
