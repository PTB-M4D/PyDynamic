import pytest
from hypothesis import given, HealthCheck, settings

from PyDynamic.signals import Signal
from test.test_signal_class.conftest import signal_inputs


@given(signal_inputs())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)
@pytest.mark.slow
def test_signal_raise_value_error_on_non_matching_dimension_of_time_and_values(inputs):
    inputs["time"] = inputs["time"][:-1]
    with pytest.raises(
        ValueError,
        match=r"Signal: Number of elements of the provided time and signal vectors "
        "are expected to match, but time is of length.*",
    ):
        Signal(**inputs)


@given(signal_inputs(ensure_uncertainty_array=True))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)
@pytest.mark.slow
def test_signal_raise_value_error_on_non_matching_dimension_of_uncertainties(inputs):
    inputs["uncertainty"] = inputs["uncertainty"][:-1]
    with pytest.raises(
        ValueError,
        match=r"Signal: if uncertainties are provided as np.ndarray "
        r"they are expected to match the number of elements of the "
        r"provided time vector, but uncertainties are of shape.*",
    ):
        Signal(**inputs)
