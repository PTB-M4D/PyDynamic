import hypothesis.strategies as hst
from hypothesis import given, HealthCheck, settings

# noinspection PyProtectedMember
from PyDynamic.model_estimation.fit_filter import (
    LSFIR,
)
from ..conftest import (
    hypothesis_dimension,
)


@given(hypothesis_dimension(min_value=4, max_value=8), hst.booleans())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
    ],
)
def test_usual_call_LSFIR_with_None_uncertainties(
    monte_carlo, freqs, sampling_freq, filter_order, fit_reciprocal
):
    LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        f=freqs,
        Fs=sampling_freq,
        tau=filter_order // 2,
        inv=fit_reciprocal,
        UH=None,
    )
