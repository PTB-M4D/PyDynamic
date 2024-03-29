import hypothesis.strategies as hst
from hypothesis import given, HealthCheck, settings

# noinspection PyProtectedMember
from PyDynamic.model_estimation.fit_filter import (
    LSFIR,
)
from ..conftest import (
    _is_np_array,
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
def test(monte_carlo, freqs, sampling_freq, filter_order, fit_reciprocal):
    b, Ub = LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        f=freqs,
        Fs=sampling_freq,
        tau=filter_order // 2,
        inv=fit_reciprocal,
        UH=None,
    )
    assert _is_np_array(b) and len(b) == filter_order + 1
    assert Ub is None
