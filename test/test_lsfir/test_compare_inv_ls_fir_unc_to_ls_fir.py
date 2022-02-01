import pytest
from hypothesis import given, HealthCheck, settings
from numpy.testing import assert_allclose

# noinspection PyProtectedMember
from PyDynamic.model_estimation.fit_filter import (
    invLSFIR_unc,
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
def test(monte_carlo, freqs, sampling_freq, filter_order):
    b_fir_mc, Ub_fir_mc = invLSFIR_unc(
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
    )
    assert_allclose(b_fir_mc, b_fir)
    assert_allclose(Ub_fir_mc, Ub_fir, atol=6e-1, rtol=6e-1)
