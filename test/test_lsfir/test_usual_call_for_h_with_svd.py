import pytest
from hypothesis import given, HealthCheck, settings

# noinspection PyProtectedMember
from PyDynamic.model_estimation.fit_filter import (
    LSFIR,
)
from .conftest import weights
from ..conftest import (
    _print_during_test_to_avoid_timeout,
    hypothesis_dimension,
)


@given(hypothesis_dimension(min_value=4, max_value=8), weights())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
    ],
)
@pytest.mark.slow
def test_usual_call_LSFIR_for_fitting_H_directly_with_svd(
    capsys, monte_carlo, freqs, sampling_freq, filter_order, weight_vector
):
    LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        f=freqs,
        Fs=sampling_freq,
        tau=filter_order // 2,
        weights=weight_vector,
        inv=True,
        UH=monte_carlo["UH"],
    )
    _print_during_test_to_avoid_timeout(capsys)
