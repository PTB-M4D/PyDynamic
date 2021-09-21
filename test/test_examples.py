from matplotlib import pyplot

from PyDynamic.examples.uncertainty_for_dft.deconv_DFT import DftDeconvolutionExample


def test_deconvolution_example(monkeypatch):
    # Test executability of the deconvolution example.
    # With this expression we override the matplotlib.pyplot.show method with a
    # lambda expression returning None but only for this one test.
    monkeypatch.setattr(pyplot, "show", lambda: None, raising=True)
    DftDeconvolutionExample()
