import hypothesis.strategies as hst
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite
from numpy.random import default_rng

from examples.demonstrate_fit_som import demonstrate_second_order_model_fitting
from PyDynamic import fit_som, make_semiposdef, sos_FreqResp
from .conftest import hypothesis_bounded_float


@pytest.mark.slow
def test_demonstrate_second_order_model_fitting(monkeypatch):
    # With this expression we override the matplotlib.pyplot.show method with a
    # lambda expression returning None but only for this one test.
    monkeypatch.setattr(plt, "show", lambda: None, raising=True)
    demonstrate_second_order_model_fitting()


@composite
def random_input_to_fit_som(draw):
    rng = default_rng()
    # sensor/measurement system
    S0 = draw(hypothesis_bounded_float(min_value=0.124, max_value=0.124))
    uS0 = draw(hypothesis_bounded_float(min_value=1e-4, max_value=1e-4))
    delta = draw(hypothesis_bounded_float(min_value=1e-2, max_value=1e-2))
    udelta = draw(hypothesis_bounded_float(min_value=1e-3, max_value=1e-3))
    f0 = draw(hst.integers(min_value=36, max_value=36))
    uf0 = draw(hypothesis_bounded_float(min_value=0.5, max_value=0.5))

    # Monte Carlo for calculation of unc. assoc. with [real(H),imag(H)]
    MCruns = draw(hst.sampled_from((1000, 2000, 4000)))
    white_nois_S0s = rng.normal(loc=S0, scale=uS0, size=MCruns)
    white_noise_deltas = rng.normal(loc=delta, scale=udelta, size=MCruns)
    white_nois_f0s = rng.normal(loc=f0, scale=uf0, size=MCruns)
    frequencies = np.linspace(0, 1.2 * f0, 30)

    HMC = sos_FreqResp(white_nois_S0s, white_noise_deltas, white_nois_f0s, frequencies)

    H_complex = np.mean(HMC, dtype=complex, axis=1)
    H = np.r_[np.real(H_complex), np.imag(H_complex)]
    UH = make_semiposdef(
        np.cov(np.r_[np.real(HMC), np.imag(HMC)], rowvar=True), maxiter=1000
    )

    return {"f": frequencies, "H": H, "UH": UH, "MCruns": MCruns, "scaling": 1}


@given(random_input_to_fit_som())
@settings(deadline=None)
def test_usual_calls_fit_som(params):
    assert fit_som(**params)


@given(random_input_to_fit_som())
def test_fit_som_with_too_short_f(params):
    params["f"] = params["f"][1:]
    with pytest.raises(ValueError):
        fit_som(**params)


@given(random_input_to_fit_som())
def test_fit_som_with_too_short_H(params):
    params["H"] = params["H"][1:]
    with pytest.raises(ValueError):
        fit_som(**params)


@given(random_input_to_fit_som())
def test_fit_som_with_too_short_UH(params):
    params["UH"] = params["UH"][1:]
    with pytest.raises(ValueError):
        fit_som(**params)


@given(random_input_to_fit_som())
def test_fit_som_with_unsquare_UH(params):
    params["UH"] = params["UH"][:,1:]
    with pytest.raises(ValueError):
        fit_som(**params)


@given(random_input_to_fit_som())
def test_fit_som_with_nonint_MCruns(params):
    params["MCruns"] = float(params["MCruns"])
    with pytest.raises(ValueError):
        fit_som(**params)
