"""
Perform test for uncertainty.propagate_filter
"""

import numpy as np

from PyDynamic.misc.filterstuff import kaiser_lowpass
from PyDynamic.misc.noise import power_law_acf, power_law_noise, white_gaussian
from PyDynamic.misc.testsignals import rect
from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.uncertainty.propagate_filter import (
    FIRuncFilter,
    IIRuncFilter,
)

# parameters of simulated measurement
Fs = 100e3  # sampling frequency (in Hz)
Ts = 1 / Fs  # sampling interval length (in s)

# nominal system parameters
fcut = 20e3  # low-pass filter cut-off frequency (6 dB)
L = 100  # filter order
b1 = kaiser_lowpass(L, fcut, Fs)[0]
b2 = kaiser_lowpass(L - 20, fcut, Fs)[0]

# uncertain knowledge: cutoff between 19.5kHz and 20.5kHz
runs = 1000
FC = fcut + (2 * np.random.rand(runs) - 1) * 0.5e3

B = np.zeros((runs, L + 1))
for k in range(runs):  # Monte Carlo for filter coefficients of low-pass filter
    B[k, :] = kaiser_lowpass(L, FC[k], Fs)[0]

Ub = make_semiposdef(np.cov(B, rowvar=0))  # covariance matrix of MC result

# simulate input and output signals
nTime = 500
time = np.arange(nTime) * Ts  # time values

# different cases
sigma_noise = 1e-2  # 1e-5


def test_FIRuncFilter_float():

    # input signal + run methods
    x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=sigma_noise)

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
    x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=noise)

    # apply uncertain FIR filter (GUM formula)
    for blow in [None, b2]:
        y, Uy = FIRuncFilter(x, Ux, b1, Ub, blow=blow, kind="corr")
        assert len(y) == len(x)
        assert len(Uy) == len(x)


def test_FIRuncFilter_diag():
    sigma_diag = sigma_noise * (
        1 + np.heaviside(np.arange(len(time)) - len(time) // 2, 0)
    )  # std doubles after half of the time
    noise = sigma_diag * white_gaussian(len(time))

    # input signal + run methods
    x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=noise)

    # apply uncertain FIR filter (GUM formula)
    for blow in [None, b2]:
        y, Uy = FIRuncFilter(x, sigma_diag, b1, Ub, blow=blow, kind="diag")
        assert len(y) == len(x)
        assert len(Uy) == len(x)


def test_IIRuncFilter():
    # define filter
    b = np.array([0.01967691, -0.01714282, 0.03329653, -0.01714282, 0.01967691])
    a = np.array([1.0, -3.03302405, 3.81183153, -2.29112937, 0.5553678])

    # simulate input and output signals
    Fs = 100e3  # sampling frequency (in Hz)
    Ts = 1 / Fs  # sampling interval length (in s)
    nx = 500
    time = np.arange(nx) * Ts  # time values

    # input signal + run methods
    sigma_noise = 1e-2  # std for input signal
    x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=sigma_noise)  # generate input signal
    Ux = sigma_noise * np.ones_like(x)  # uncertainty of input signal
    Uab = np.diag(np.zeros((len(a) + len(b) - 1)))  # uncertainty of IIR-parameters
    Uab[2, 2] = 0.000001  # only a2 is uncertain

    # run x all at once
    y, Uy, _ = IIRuncFilter(x, Ux, b, a, Uab=Uab, kind="diag")
    assert len(y) == len(x)
    assert len(Uy) == len(x)

    # slice x into smaller chunks and process them in batches
    # this tests the internal state options
    y_list = []
    Uy_list = []
    state = None
    for x_batch, Ux_batch in zip(np.array_split(x, 200), np.array_split(Ux, 200)):
        yi, Uyi, state = IIRuncFilter(
            x_batch, Ux_batch, b, a, Uab=Uab, kind="diag", state=state
        )
        y_list.append(yi)
        Uy_list.append(Uyi)
    yb = np.concatenate(y_list, axis=0)
    Uyb = np.concatenate(Uy_list, axis=0)
    assert len(yb) == len(x)
    assert len(Uyb) == len(x)

    # check if both ways of calling IIRuncFilter yield the same result
    assert np.allclose(yb, y)
    assert np.allclose(Uyb, Uy)


def test_FIR_IIR_identity():

    # define filter
    b = np.array([0.01967691, -0.01714282, 0.03329653, -0.01714282, 0.01967691])
    a = np.array([1.0])

    # simulate input and output signals
    Fs = 100e3  # sampling frequency (in Hz)
    Ts = 1 / Fs  # sampling interval length (in s)
    nx = 500
    time = np.arange(nx) * Ts  # time values

    # input signal + run methods
    sigma_noise = 1e-2  # std for input signal
    x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=sigma_noise)  # generate input signal
    Ux = sigma_noise * np.ones_like(x)  # uncertainty/autocorrelation of input signal
    Uab = np.diag(
        np.zeros((len(a) + len(b) - 1))
    )  # fully certain filter-parameters, otherwise FIR and IIR do not match! (see docstring of IIRuncFilter)

    for kind in ["diag", "corr"]:
        # run signal through both implementations
        y_iir, Uy_iir, _ = IIRuncFilter(x, Ux, b, a, Uab=Uab, kind=kind)
        y_fir, Uy_fir = FIRuncFilter(x, Ux, theta=b, Utheta=Uab, kind=kind)

        assert np.allclose(y_fir, y_iir)
        assert np.allclose(Uy_fir[len(b) :], Uy_iir[len(b) :])
