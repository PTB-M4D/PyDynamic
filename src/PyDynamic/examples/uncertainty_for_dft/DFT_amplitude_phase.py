"""Demonstrate the transformation from real-imag to amplitude-phase and back

We use the corresponding PyDynamic functions from the module
:mod:`PyDynamic.uncertainty.propagate_DFT´.

The function Time2AmpPhase uses GUM_DFT and DFT2AmpPhase internally.
The function AmpPhase2Time applies the formulas as given in the corresponding
publication on GUM2DFT.
"""

import matplotlib.pyplot as plt
import numpy as np

from PyDynamic.misc.testsignals import multi_sine
from PyDynamic.uncertainty.propagate_DFT import (
    AmpPhase2Time,
    GUM_DFTfreq,
    Time2AmpPhase,
    Time2AmpPhase_multi,
)


class DftAmplitudePhaseExample:
    def __init__(self):
        multi_sine_amps = np.random.randint(1, 4, 10)
        multi_sine_freqs = np.linspace(100, 500, len(multi_sine_amps)) * 2 * np.pi
        time_step = 0.0001
        time = np.arange(start=0.0, stop=0.2, step=time_step)
        measure_noise_std_sigma = 0.001
        testsignal = multi_sine(
            time, multi_sine_amps, multi_sine_freqs, noise=measure_noise_std_sigma
        )

        plt.figure(1, figsize=(12, 6))
        plt.plot(time, testsignal)
        plt.xlabel("time in s")
        plt.ylabel("signal amplitude in a.u.")

        # uncertainty propagation from time domain to frequency domain
        A, P, UAP = Time2AmpPhase(testsignal, measure_noise_std_sigma ** 2)
        f = GUM_DFTfreq(len(time), time_step)

        plt.figure(2, figsize=(12, 6))
        plt.errorbar(f, A, np.sqrt(np.diag(UAP)[: len(A)]), fmt=".-")
        plt.xlabel("frequency in Hz")
        plt.ylabel("DFT magnitude values in a.u.")

        # uncertainty propagation from frequency domain to time domain
        x, ux = AmpPhase2Time(A, P, UAP)

        plt.figure(3, figsize=(12, 6))
        plt.subplot(211)
        plt.errorbar(
            time,
            x,
            np.sqrt(np.diag(ux)),
            fmt=".-",
            label="after DFT and iDFT using amplitude and phase",
        )
        plt.plot(time, testsignal, label="original signal")
        plt.xlabel("time in s")
        plt.ylabel("signal amplitude in a.u.")
        plt.legend()
        plt.subplot(212)
        plt.plot(
            time,
            np.sqrt(np.diag(ux)),
            label="uncertainty after DFT and iDFT using amplitude and phase",
        )
        plt.plot(
            time,
            np.ones_like(time) * measure_noise_std_sigma,
            label="original uncertainty",
        )
        plt.xlabel("time in s")
        plt.ylabel("uncertainty in a.u.")
        plt.legend()

        # # apply the same method to several signals with one call
        M = 10
        testsignals = np.zeros((M, len(time)))
        for m in range(M):
            testsignals[m, :] = multi_sine(
                time, multi_sine_amps, multi_sine_freqs, noise=measure_noise_std_sigma
            )

        # select those frequencies of the 10 % largest magnitude values
        indices = np.argsort(A)[: len(A) // 10]
        # propagate from time to frequency domain and select specified frequencies
        A_multi, P_multi, UAP_multi = Time2AmpPhase_multi(
            testsignals, np.ones(M) * measure_noise_std_sigma ** 2, selector=indices
        )

        plt.figure(4, figsize=(12, 6))
        plt.subplot(211)
        plt.plot(f[indices], A_multi.T, linestyle=":")
        plt.subplot(212)
        plt.plot(f[indices], P_multi.T, linestyle=":")


if __name__ == "__main__":
    DftAmplitudePhaseExample()
