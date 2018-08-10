# -*- coding: utf-8 -*-
"""
Perform tests on identification sub-packages.

"""

import numpy as np

from PyDynamic.identification import fit_filter
from PyDynamic.misc.SecondOrderSystem import sos_FreqResp


def test_LSIIR():
    # measurement system
    f0 = 36e3  # system resonance frequency in Hz
    S0 = 0.124  # system static gain
    delta = 0.0055  # system damping

    f = np.linspace(0, 80e3, 30)  # frequencies for fitting the system
    Hvals = sos_FreqResp(S0, delta, f0, f)  # frequency response of the 2nd order system

    # %% fitting the IIR filter

    Fs = 500e3  # sampling frequency
    Na = 4
    Nb = 4  # IIR filter order (Na - denominator, Nb - numerator)

    b, a, tau = fit_filter.LSIIR(Hvals, Na, Nb, f, Fs)

    assert len(b) == Nb + 1
    assert len(a) == Na + 1
    assert isinstance(tau, int)
