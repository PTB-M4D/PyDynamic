import pytest
from matplotlib import pyplot

from PyDynamic.examples.uncertainty_for_dft.deconv_DFT import DftDeconvolutionExample
from PyDynamic.examples.uncertainty_for_dft.DFT_amplitude_phase import (
    DftAmplitudePhaseExample,
)


@pytest.mark.slow
def test_deconvolution_example(monkeypatch):
    monkeypatch.setattr(pyplot, "show", lambda: None, raising=True)
    DftDeconvolutionExample()


@pytest.mark.slow
def test_dft_amp_phase_example(monkeypatch):
    monkeypatch.setattr(pyplot, "show", lambda: None, raising=True)
    DftAmplitudePhaseExample()
