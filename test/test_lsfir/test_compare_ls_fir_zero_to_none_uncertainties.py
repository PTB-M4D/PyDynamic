import numpy as np
import pytest
from hypothesis import given, HealthCheck, settings
from numpy.testing import assert_allclose

# noinspection PyProtectedMember
from PyDynamic.model_estimation.fit_filter import (
    LSFIR,
)
from ..conftest import (
    hypothesis_dimension,
)


@given(hypothesis_dimension(min_value=4, max_value=8))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
    ],
)
@pytest.mark.slow
def test_compare_LSFIR_with_zero_to_None_uncertainties_with_svd_for_fitting_one_over_H(
    monte_carlo, freqs, sampling_freq, filter_order
):
    b_fir_svd = LSFIR(
        H=monte_carlo["H"],
        UH=np.zeros_like(monte_carlo["UH"]),
        N=filter_order,
        tau=filter_order // 2,
        f=freqs,
        Fs=sampling_freq,
        inv=True,
    )[0]
    b_fir_none = LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        tau=filter_order // 2,
        f=freqs,
        Fs=sampling_freq,
        inv=True,
        UH=None,
    )[0]
    assert_allclose(b_fir_svd, b_fir_none)


@given(hypothesis_dimension(min_value=4, max_value=8))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
    ],
)
@pytest.mark.slow
def test_compare_LSFIR_with_zero_to_None_uncertainties_and_mc_for_fitting_one_over_H(
    monte_carlo, freqs, sampling_freq, filter_order
):
    b_fir_mc = LSFIR(
        H=monte_carlo["H"],
        UH=np.zeros_like(monte_carlo["UH"]),
        N=filter_order,
        tau=filter_order // 2,
        f=freqs,
        Fs=sampling_freq,
        inv=True,
        mc_runs=2,
    )[0]
    b_fir_none = LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        tau=filter_order // 2,
        f=freqs,
        Fs=sampling_freq,
        inv=True,
        UH=None,
    )[0]
    assert_allclose(b_fir_mc, b_fir_none)


@given(hypothesis_dimension(min_value=4, max_value=8))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
    ],
)
@pytest.mark.slow
def test_compare_LSFIR_with_zero_to_None_uncertainties_and_mc_for_fitting_H_directly(
    monte_carlo, freqs, sampling_freq, filter_order
):
    b_fir_mc = LSFIR(
        H=monte_carlo["H"],
        UH=np.zeros_like(monte_carlo["UH"]),
        N=filter_order,
        tau=filter_order // 2,
        f=freqs,
        Fs=sampling_freq,
        inv=False,
        mc_runs=2,
    )[0]
    b_fir_none = LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        tau=filter_order // 2,
        f=freqs,
        Fs=sampling_freq,
        inv=False,
        UH=None,
    )[0]
    assert_allclose(b_fir_mc, b_fir_none)
