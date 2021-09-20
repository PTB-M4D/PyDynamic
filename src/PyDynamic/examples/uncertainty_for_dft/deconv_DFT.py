"""Example for the propagation of uncertainties for deconvolution in the Fourier domain

In this example an actually measured signal is deconvolved and the result compared to a
reference measurement signal.

Created on Thu Sep 10 11:58:46 2015
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from PyDynamic.uncertainty.propagate_DFT import (
    AmpPhase2DFT,
    DFT_deconv,
    DFT_multiply,
    GUM_DFT,
    GUM_iDFT,
)


class DftDeconvolutionExample:
    def __init__(self):
        self.file_directory = os.path.dirname(os.path.abspath(__file__))
        self.colors = [[0.1, 0.6, 0.5], [0.9, 0.2, 0.5], [0.9, 0.5, 0.1]]

        # low-pass filter for deconvolution
        def lowpass(f, fcut=80e6):
            return 1 / (1 + 1j * f / fcut) ** 2

        #%% reference data
        ref_file = np.loadtxt(
            os.path.join(self.file_directory, "DFTdeconv reference_signal.dat")
        )
        self.time = ref_file[:, 0]
        self.ref_data = ref_file[:, 1]

        N = len(self.time)

        #%% measured hydrophone output signal
        meas = np.loadtxt(
            os.path.join(self.file_directory, "DFTdeconv measured_signal.dat")
        )
        y = meas[:, 1]
        # assumed noise std
        noise_std = 4e-4
        Uy = noise_std ** 2

        #%% hydrophone calibration data
        calib = np.loadtxt(
            os.path.join(self.file_directory, "DFTdeconv calibration.dat")
        )
        self.f = calib[:, 0]
        self.FR = calib[:, 1] * np.exp(1j * calib[:, 3])
        Nf = 2 * (len(self.f) - 1)

        uAmp = calib[:, 2]
        uPhas = calib[:, 4]
        self.UAP = np.r_[uAmp, uPhas * np.pi / 180] ** 2

        print("Propagating uncertainty associated with measurement through DFT")
        Y, UY = GUM_DFT(y, Uy, N=Nf)
        # propagation to real/imag
        print(
            "Propagating uncertainty associated with calibration data to real and imag "
            "part"
        )
        H, UH = AmpPhase2DFT(np.abs(self.FR), np.angle(self.FR), self.UAP)
        # propagation through deconvolution equation
        print("Propagating uncertainty through the inverse system")
        XH, UXH = DFT_deconv(H, Y, UH, UY)
        #%% low pass filter for deconvolution regularization
        self.HLc = lowpass(self.f)
        HL = np.r_[np.real(self.HLc), np.imag(self.HLc)]
        # application of low-pass filter
        print("Propagating uncertainty through the low-pass filter")
        XH, UXH = DFT_multiply(XH, HL, UXH)
        # propagation back to time domain
        print(
            "Propagating uncertainty associated with the estimate back to time domain\n"
        )
        self.xh, Uxh = GUM_iDFT(XH, UXH, Nx=N)
        self.ux = np.sqrt(np.diag(Uxh))

        self.create_figure_1_left()
        self.create_figure_1_right()
        self.create_figure_2_left()
        self.create_figure_2_right()
        self.show_plots()

    def create_figure_2_left(self):
        plt.figure(1)
        plt.clf()
        plt.plot(
            self.time * 1e6,
            self.xh,
            label="estimated pressure signal",
            linewidth=2,
            color=self.colors[0],
        )
        plt.plot(
            self.time * 1e6,
            self.ref_data,
            "--",
            label="reference data",
            linewidth=2,
            color=self.colors[1],
        )
        plt.fill_between(
            self.time * 1e6,
            self.xh + 2 * self.ux,
            self.xh - 2 * self.ux,
            alpha=0.2,
            color=self.colors[0],
        )
        plt.xlabel(r"time / $\mu$s", fontsize=22)
        plt.ylabel("signal amplitude / MPa", fontsize=22)
        plt.tick_params(which="major", labelsize=18)
        plt.legend(loc="best", fontsize=18, fancybox=True)
        plt.xlim(0, 2)

    def create_figure_1_right(self):
        def compute_db(frequency_response_amplitude):
            return 20 * np.log10(np.abs(frequency_response_amplitude))

        plt.figure(2)
        plt.clf()
        plt.plot(
            self.f * 1e-6, compute_db(self.FR), label="measurement system", linewidth=2
        )
        plt.plot(
            self.f * 1e-6,
            compute_db(self.HLc / self.FR),
            label="inverse system",
            linewidth=2,
        )
        plt.plot(self.f * 1e-6, compute_db(self.HLc), "--", linewidth=0.5)
        plt.legend(loc="best", fontsize=18)
        plt.xlim(0.5, 100)
        plt.ylim(-40, 40)
        plt.xlabel("frequency / MHz", fontsize=22)
        plt.ylabel("frequency response amplitude / dB", fontsize=22)
        plt.tick_params(which="both", labelsize=18)

    def create_figure_1_left(self):
        plt.figure(3)
        plt.clf()
        plt.subplot(211)
        plt.errorbar(
            self.f * 1e-6,
            np.abs(self.FR),
            2 * np.sqrt(self.UAP[: len(self.UAP) // 2]),
            fmt=".-",
            alpha=0.2,
            color=self.colors[0],
        )
        plt.xlim(0.5, 80)
        plt.ylim(0.04, 0.24)
        plt.xlabel("frequency / MHz", fontsize=22)
        plt.tick_params(which="both", labelsize=18)
        plt.ylabel("amplitude / V/MPa", fontsize=22)
        plt.subplot(212)
        plt.errorbar(
            self.f * 1e-6,
            np.unwrap(np.angle(self.FR)) * np.pi / 180,
            2 * self.UAP[len(self.UAP) // 2 :],
            fmt=".-",
            alpha=0.2,
            color=self.colors[0],
        )
        plt.xlim(0.5, 80)
        plt.ylim(-0.2, 0.3)
        plt.xlabel("frequency / MHz", fontsize=22)
        plt.tick_params(which="both", labelsize=18)
        plt.ylabel("phase / rad", fontsize=22)

    def create_figure_2_right(self):
        plt.figure(4)
        plt.clf()
        plt.plot(
            self.time * 1e6,
            self.ux,
            label="uncertainty",
            linewidth=2,
            color=self.colors[0],
        )
        plt.xlabel(r"time / $\mu$s", fontsize=22)
        plt.ylabel("uncertainty / MPa", fontsize=22)
        plt.tick_params(which="major", labelsize=18)
        plt.xlim(0, 2)

    @staticmethod
    def show_plots():
        plt.show()


if __name__ == "__main__":
    DftDeconvolutionExample()
