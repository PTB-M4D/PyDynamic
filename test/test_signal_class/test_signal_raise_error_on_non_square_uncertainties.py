import pytest
from hypothesis import given, HealthCheck, settings

from PyDynamic.signals import Signal
from test.test_signal_class.conftest import signal_inputs


@given(signal_inputs(ensure_uncertainty_covariance_matrix=True))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
        HealthCheck.data_too_large
    ],
)
@pytest.mark.slow
def test(inputs):
    inputs["uncertainty"] = inputs["uncertainty"][..., :-1]
    with pytest.raises(
        ValueError,
        match=r"Signal: if uncertainties are provided as 2-dimensional np.ndarray "
        r"they are expected to resemble a square matrix, but uncertainties are of "
        r"shape.*",
    ):
        Signal(**inputs)
