import numpy as np
import pytest

from PyDynamic import sos_FreqResp


@pytest.fixture
def LSIIR_parameters():
    """Design a sample measurement system and a corresponding frequqency response."""
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
    return {
        "Hvals": Hvals,
        "Na": Na,
        "Nb": Nb,
        "f": f,
        "Fs": Fs,
    }
