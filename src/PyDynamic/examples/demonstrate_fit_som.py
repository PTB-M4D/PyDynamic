"""Example for the fit of a second-order transfer function to frequency responses

We fit a second-order transfer function of the mass-damper-spring model to a set of
frequency response values
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

from PyDynamic.misc.SecondOrderSystem import sos_FreqResp
from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.model_estimation.fit_transfer import fit_som


def demonstrate_second_order_model_fitting(runs: int = 10000):
    rng = default_rng(1)

    # sensor/measurement system
    S0 = 0.124
    uS0 = 0.001
    delta = 0.01
    udelta = 0.001
    f0 = 36
    uf0 = 0.5

    # Monte Carlo for calculation of unc. assoc. with [real(H),imag(H)]
    MCS0 = rng.normal(loc=S0, scale=uS0, size=runs)
    MCd = rng.normal(loc=delta, scale=udelta, size=runs)
    MCf0 = rng.normal(loc=f0, scale=uf0, size=runs)
    f = np.linspace(0, 1.2 * f0, 30)

    HMC = sos_FreqResp(MCS0, MCd, MCf0, f)

    Hc = np.mean(HMC, dtype=complex, axis=1)
    H = np.r_[np.real(Hc), np.imag(Hc)]
    UH = np.cov(np.r_[np.real(HMC), np.imag(HMC)], rowvar=True)
    UH = make_semiposdef(UH)

    p, Up = fit_som(f, H, UH, scaling=1, MCruns=runs, verbose=True)

    plt.figure(1)
    plt.errorbar(
        range(1, 4),
        [p[0] * 10, p[1] * 10, p[2] / 10],
        np.sqrt(np.diag(Up)) * np.array([10, 10, 0.1]),
        fmt=".",
    )
    plt.errorbar(
        np.arange(1, 4) + 0.2,
        [S0 * 10, delta * 10, f0 / 10],
        [uS0 * 10, udelta * 10, uf0 / 10],
        fmt=".",
    )
    plt.xlim(0, 4)
    plt.xticks([1.1, 2.1, 3.1], [r"$S_0$", r"$\delta$", r"$f_0$"])
    plt.xlabel("scaled model parameters")

    plt.show()


if __name__ == "__main__":
    demonstrate_second_order_model_fitting()
