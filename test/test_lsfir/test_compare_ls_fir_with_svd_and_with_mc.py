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
    ],
)
@pytest.mark.slow
def test(monte_carlo, freqs, sampling_freq, filter_order):
    b_fir_svd, Ub_fir_svd = LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        f=freqs,
        Fs=sampling_freq,
        tau=filter_order // 2,
        verbose=True,
        inv=True,
        UH=monte_carlo["UH"],
    )
    b_fir_mc, Ub_fir_mc = LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        f=freqs,
        Fs=sampling_freq,
        tau=filter_order // 2,
        verbose=True,
        inv=True,
        UH=monte_carlo["UH"],
        mc_runs=10000,
    )
    assert_allclose(b_fir_mc, b_fir_svd, rtol=9e-2)
    assert_allclose(Ub_fir_mc, Ub_fir_svd, atol=6e-1, rtol=6e-1)
