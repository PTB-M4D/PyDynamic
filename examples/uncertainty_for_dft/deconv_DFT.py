# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:58:46 2015

In this example an actually measured signal is deconvolved and the result compared to a reference measurement signal.

"""
import os

import numpy as np
import matplotlib.pyplot as plt
from PyDynamic.uncertainty.propagate_DFT import (
    GUM_DFT,
    GUM_iDFT,
    DFT_deconv,
    AmpPhase2DFT,
    DFT_multiply,
)


def demonstrate_deconvolution():
    file_directory = os.path.dirname(os.path.abspath(__file__))
    colors = [[0.1, 0.6, 0.5], [0.9, 0.2, 0.5], [0.9, 0.5, 0.1]]

    # low-pass filter for deconvolution
    def lowpass(f, fcut=80e6):
        return 1 / (1 + 1j * f / fcut) ** 2

    #%% reference data
    ref_file = np.loadtxt(
        os.path.join(file_directory, "DFTdeconv reference_signal.dat")
    )
    time = ref_file[:, 0]
    ref_data = ref_file[:, 1]
    Ts = 2e-9
    N = len(time)

    #%% measured hydrophone output signal
    meas = np.loadtxt(
        os.path.join(file_directory, "DFTdeconv measured_signal.dat")
    )
    y = meas[:, 1]
    # assumed noise std
    noise_std = 4e-4
    Uy = noise_std ** 2

    #%% hydrophone calibration data
    calib = np.loadtxt(
        os.path.join(file_directory, "DFTdeconv calibration.dat")
    )
    f = calib[:, 0]
    FR = calib[:, 1] * np.exp(1j * calib[:, 3])
    Nf = 2 * (len(f) - 1)

    uAmp = calib[:, 2]
    uPhas = calib[:, 4]
    UAP = np.r_[uAmp, uPhas * np.pi / 180] ** 2

    print("Propagating uncertainty associated with measurement through DFT")
    Y, UY = GUM_DFT(y, Uy, N=Nf)
    # propagation to real/imag
    print(
        "Propagating uncertainty associated with calibration data to real and imag part"
    )
    H, UH = AmpPhase2DFT(np.abs(FR), np.angle(FR), UAP)
    # propagation through deconvolution equation
    print("Propagating uncertainty through the inverse system")
    XH, UXH = DFT_deconv(H, Y, UH, UY)
    #%% low pass filter for deconvolution regularization
    HLc = lowpass(f)
    HL = np.r_[np.real(HLc), np.imag(HLc)]
    # application of low-pass filter
    print("Propagating uncertainty through the low-pass filter")
    XH, UXH = DFT_multiply(XH, HL, UXH)
    # propagation back to time domain
    print("Propagating uncertainty associated with the estimate back to time domain\n")
    xh, Uxh = GUM_iDFT(XH, UXH, Nx=N)
    ux = np.sqrt(np.diag(Uxh))

    #%% plotting results
    plt.figure(1)
    plt.clf()
    plt.plot(
        time * 1e6, xh, label="estimated pressure signal", linewidth=2, color=colors[0]
    )
    plt.plot(
        time * 1e6, ref_data, "--", label="reference data", linewidth=2, color=colors[1]
    )
    plt.fill_between(time * 1e6, xh + 2 * ux, xh - 2 * ux, alpha=0.2, color=colors[0])
    plt.xlabel(r"time / $\mu$s", fontsize=22)
    plt.ylabel("signal amplitude / MPa", fontsize=22)
    plt.tick_params(which="major", labelsize=18)
    plt.legend(loc="best", fontsize=18, fancybox=True)
    plt.xlim(0, 2)

    dB = lambda a: 20 * np.log10(np.abs(a))
    plt.figure(2)
    plt.clf()
    plt.plot(f * 1e-6, dB(FR), label="measurement system", linewidth=2)
    plt.plot(f * 1e-6, dB(HLc / FR), label="inverse system", linewidth=2)
    plt.plot(f * 1e-6, dB(HLc), "--", linewidth=0.5)
    plt.legend(loc="best", fontsize=18)
    plt.xlim(0.5, 100)
    plt.ylim(-40, 40)
    plt.xlabel("frequency / MHz", fontsize=22)
    plt.ylabel("frequency response amplitude / dB", fontsize=22)
    plt.tick_params(which="both", labelsize=18)

    plt.figure(3)
    plt.clf()
    plt.subplot(211)
    plt.errorbar(
        f * 1e-6,
        np.abs(FR),
        2 * np.sqrt(UAP[: len(UAP) // 2]),
        fmt=".-",
        alpha=0.2,
        color=colors[0],
    )
    plt.xlim(0.5, 80)
    plt.ylim(0.04, 0.24)
    plt.xlabel("frequency / MHz", fontsize=22)
    plt.tick_params(which="both", labelsize=18)
    plt.ylabel("amplitude / V/MPa", fontsize=22)
    plt.subplot(212)
    plt.errorbar(
        f * 1e-6,
        np.unwrap(np.angle(FR)) * np.pi / 180,
        2 * UAP[len(UAP) // 2 :],
        fmt=".-",
        alpha=0.2,
        color=colors[0],
    )
    plt.xlim(0.5, 80)
    plt.ylim(-0.2, 0.3)
    plt.xlabel("frequency / MHz", fontsize=22)
    plt.tick_params(which="both", labelsize=18)
    plt.ylabel("phase / rad", fontsize=22)

    plt.figure(4)
    plt.clf()
    plt.plot(time * 1e6, ux, label="uncertainty", linewidth=2, color=colors[0])
    plt.xlabel(r"time / $\mu$s", fontsize=22)
    plt.ylabel("uncertainty / MPa", fontsize=22)
    plt.tick_params(which="major", labelsize=18)
    plt.xlim(0, 2)

    plt.show()


if __name__ == "__main__":
    demonstrate_deconvolution()
