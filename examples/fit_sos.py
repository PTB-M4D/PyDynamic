# -*- coding: utf-8 -*-
"""
Example for the fit of a second-order transfer function (mass-damper-spring model)
to a set of frequency response values.
"""

from matplotlib.pyplot import *

from PyDynamic.model_estimation.fit_transfer import fit_som
from PyDynamic.misc.SecondOrderSystem import sos_FreqResp
from PyDynamic.misc.tools import make_semiposdef

rst = np.random.RandomState(1)

# sensor/measurement system
S0 = 0.124
uS0 = 0.001
delta = 0.01
udelta = 0.001
f0 = 36
uf0 = 0.5
f = np.linspace(0, f0 * 1.2, 20)

# Monte Carlo for calculation of unc. assoc. with [real(H),imag(H)]
runs = 10000
MCS0 = S0 + rst.randn(runs) * uS0
MCd = delta + rst.randn(runs) * udelta
MCf0 = f0 + rst.randn(runs) * uf0
f = np.linspace(0, 1.2 * f0, 30)

HMC = sos_FreqResp(MCS0, MCd, MCf0, f)

Hc = np.mean(HMC, dtype=complex, axis=1)
H = np.r_[np.real(Hc), np.imag(Hc)]
UH = np.cov(np.r_[np.real(HMC), np.imag(HMC)], rowvar=True)
UH = make_semiposdef(UH)

p, Up, HMC2 = fit_som(f, Hc, UH, scaling=1, MCruns=10000)

figure(1)
errorbar(
    range(1, 4),
    [p[0] * 10, p[1] * 10, p[2] / 10],
    np.sqrt(np.diag(Up)) * np.array([10, 10, 0.1]),
    fmt=".",
)
errorbar(
    np.arange(1, 4) + 0.2,
    [S0 * 10, delta * 10, f0 / 10],
    [uS0 * 10, udelta * 10, uf0 / 10],
    fmt=".",
)
xlim(0, 4)
xticks([1.1, 2.1, 3.1], [r"$S_0$", r"$\delta$", r"$f_0$"])
xlabel("scaled model parameters")

show()
