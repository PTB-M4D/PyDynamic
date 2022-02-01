import pytest
from hypothesis import given, HealthCheck, settings
from numpy.testing import assert_allclose

# noinspection PyProtectedMember
from PyDynamic.model_estimation.fit_filter import (
    LSFIR,
)
from ..conftest import (
    _print_during_test_to_avoid_timeout,
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
def test_compare_different_dtypes_LSFIR(
    capsys,
    monte_carlo,
    complex_H_with_UH,
    freqs,
    sampling_freq,
    filter_order,
):
    b_real_imaginary, ub_real_imaginary = LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        f=freqs,
        Fs=sampling_freq,
        tau=filter_order // 2,
        inv=True,
        verbose=True,
        UH=monte_carlo["UH"],
        mc_runs=10000,
    )
    b_complex, ub_complex = LSFIR(
        H=complex_H_with_UH["H"],
        N=filter_order,
        f=freqs,
        Fs=sampling_freq,
        tau=filter_order // 2,
        inv=True,
        verbose=True,
        UH=monte_carlo["UH"],
        mc_runs=10000,
    )
    _print_during_test_to_avoid_timeout(capsys)
    assert_allclose(b_real_imaginary, b_complex, rtol=4e-2)
    assert_allclose(ub_real_imaginary, ub_complex, rtol=6e-1)
