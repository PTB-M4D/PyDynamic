""" Test the DFT example *examples/DFT and iDFT with PyDynamic...ipynb*."""

import numpy as np
from matplotlib.pyplot import (
    errorbar,
    figure,
    plot,
    subplot,
    subplots_adjust,
    xlabel,
    xlim,
    xticks,
    ylabel,
)
from numpy import fft, random, sqrt
from numpy.ma import arange, sin
from scipy.constants import pi

from PyDynamic import GUM_DFT


def test_run_copy_of_notebook_code():
    np.random.seed(123)
    Fs = 100  # sampling frequency in Hz
    Ts = 1 / Fs  # sampling interval in s
    N = 1024  # number of samples
    time = arange(0, N * Ts, Ts)  # time instants
    noise_std = 0.1  # signal noise standard deviation
    # time domain signal
    x = (
        sin(2 * pi * Fs / 10 * time)
        + sin(2 * pi * Fs / 5 * time)
        + random.randn(len(time)) * noise_std
    )

    # Apply DFT with propagation of uncertainties
    X, UX = GUM_DFT(x, noise_std ** 2)
    f = fft.rfftfreq(N, Ts)  # frequency values

    figure()
    plot(time, x)
    xlim(time[0], time[-1])
    xlabel("time / s", fontsize=18)
    ylabel("signal amplitude / au", fontsize=18)

    figure()
    subplot(211)
    errorbar(f, X[: len(f)], sqrt(UX[: len(f)]))
    ylabel("real part", fontsize=18)
    xticks([])
    subplot(212)
    errorbar(f, X[len(f) :], sqrt(UX[len(f) :]))
    ylabel("imaginary part", fontsize=18)
    xlabel("frequency / Hz", fontsize=18)
    subplots_adjust(hspace=0.05)
