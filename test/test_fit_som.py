import hypothesis.strategies as hst
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import assume, given, HealthCheck, settings, Verbosity
from hypothesis.strategies import composite
from numpy.random import default_rng

from PyDynamic import fit_som, make_semiposdef, sos_FreqResp
from PyDynamic.examples.demonstrate_fit_som import (
    demonstrate_second_order_model_fitting,
)
from .conftest import hypothesis_float_vector


def test_demonstrate_second_order_model_fitting(monkeypatch):
    # With this expression we override the matplotlib.pyplot.show method with a
    # lambda expression returning None but only for this one test.
    # guarantee this
    monkeypatch.setattr(plt, "show", lambda: None, raising=True)
    demonstrate_second_order_model_fitting(runs=100)


@composite
def random_input_to_fit_som(draw, guarantee_UH_as_matrix: bool = False):
    rng = default_rng()
    # sensor/measurement system
    S0 = 0.124
    uS0 = 1e-4
    delta = 1e-2
    udelta = 1e-3
    f0 = 36
    uf0 = 0.5

    # Monte Carlo for calculation of unc. assoc. with [real(H),imag(H)]
    MCruns = draw(hst.sampled_from((10, 20, 40)))
    white_noise_S0s = rng.normal(loc=S0, scale=uS0, size=MCruns)
    white_noise_deltas = rng.normal(loc=delta, scale=udelta, size=MCruns)
    white_noise_f0s = rng.normal(loc=f0, scale=uf0, size=MCruns)
    frequencies = np.linspace(0, 1.2 * f0, 30)

    HMC = sos_FreqResp(
        white_noise_S0s, white_noise_deltas, white_noise_f0s, frequencies
    )

    H_complex = np.mean(HMC, dtype=complex, axis=1)
    H = np.r_[np.real(H_complex), np.imag(H_complex)]
    UH = make_semiposdef(
        np.cov(np.r_[np.real(HMC), np.imag(HMC)], rowvar=True), maxiter=1000
    )
    if not guarantee_UH_as_matrix:
        UH = draw(hst.sampled_from((UH, None)))
    weighting = draw(
        hst.one_of(
            (hypothesis_float_vector(length=len(H), min_value=1e-2, max_value=1)),
            hst.sampled_from(("cov", "diag", None)),
        )
    )
    return {
        "f": frequencies,
        "H": H,
        "UH": UH,
        "MCruns": MCruns,
        "weighting": weighting,
        "scaling": 1,
    }


@pytest.mark.slow
@given(random_input_to_fit_som())
@settings(
    deadline=None,
    verbosity=Verbosity.verbose,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_usual_calls_fit_som(capsys, params):
    with capsys.disabled():
        fit_som(verbose=True, **params)


@given(random_input_to_fit_som())
def test_fit_som_with_too_short_f(params):
    params["f"] = params["f"][1:]
    with pytest.raises(ValueError):
        fit_som(**params)


@given(random_input_to_fit_som())
@pytest.mark.slow
def test_fit_som_with_too_short_H(params):
    params["H"] = params["H"][1:]
    with pytest.raises(ValueError):
        fit_som(**params)


@given(random_input_to_fit_som(guarantee_UH_as_matrix=True))
def test_fit_som_with_too_short_UH(params):
    params["UH"] = params["UH"][1:]
    with pytest.raises(ValueError):
        fit_som(**params)


@given(random_input_to_fit_som(guarantee_UH_as_matrix=True))
def test_fit_som_with_unsquare_UH(params):
    params["UH"] = params["UH"][:, 1:]
    with pytest.raises(ValueError):
        fit_som(**params)


@given(random_input_to_fit_som())
def test_fit_som_with_nonint_MCruns(params):
    params["MCruns"] = float(params["MCruns"])
    assume(params["UH"] is not None)
    with pytest.raises(ValueError):
        fit_som(**params)


@given(random_input_to_fit_som())
def test_fit_som_with_invalid_weighting_string(params):
    params["weighting"] = "something unexpected"
    with pytest.raises(ValueError):
        fit_som(**params)


@given(random_input_to_fit_som())
def test_fit_som_with_too_short_weighting_vector(params):
    params["weighting"] = params["H"][1:]
    with pytest.raises(ValueError):
        fit_som(**params)


@given(random_input_to_fit_som())
def test_fit_som_with_zero_frequency_response_but_without_MCruns_or_UH(params):
    params["H"][0] = 0.0
    assume(params["MCruns"] is None or params["UH"] is None)
    with pytest.raises(ValueError):
        fit_som(**params)
