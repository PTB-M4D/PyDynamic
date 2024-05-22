import pytest
from hypothesis import given, HealthCheck, settings
from numpy.testing import assert_allclose

# noinspection PyProtectedMember
from PyDynamic.model_estimation.fit_filter import (
    invLSFIR_uncMC,
    LSFIR,
)
from ..conftest import (
    hypothesis_dimension,
    custom_atol,
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
    b_fir_mc, Ub_fir_mc = invLSFIR_uncMC(
        H=monte_carlo["H"],
        UH=monte_carlo["UH"],
        N=filter_order,
        tau=filter_order // 2,
        f=freqs,
        Fs=sampling_freq,
    )
    b_fir, Ub_fir = LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        f=freqs,
        Fs=sampling_freq,
        tau=filter_order // 2,
        inv=True,
        UH=monte_carlo["UH"],
        mc_runs=10000,
    )
    assert_allclose(b_fir_mc, b_fir, rtol=4e-2, atol=custom_atol)
    assert_allclose(Ub_fir_mc, Ub_fir, rtol=6e-1, atol=6e-1)
