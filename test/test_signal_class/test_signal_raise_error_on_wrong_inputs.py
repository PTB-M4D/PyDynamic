import numpy as np
import pytest
from hypothesis import given, HealthCheck, settings

from PyDynamic.signals import Signal
from test.test_signal_class.conftest import signal_inputs


@given(signal_inputs())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
    ],
)
@pytest.mark.slow
def test_signal_class_raise_not_implemented_multivariate_signal(inputs):
    inputs["values"] = inputs["values"][..., np.newaxis]
    with pytest.raises(
        NotImplementedError,
        match=r"Signal: Multivariate signals are not implemented yet.",
    ):
        Signal(**inputs)


@given(signal_inputs(ensure_time_step_to_be_float=True))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
    ],
)
@pytest.mark.slow
def test_signal_raise_value_error_on_non_matching_sampling_freq_and_time_step(inputs):
    inputs["Fs"] = inputs["Ts"]
    with pytest.raises(
        ValueError,
        match=r"Signal: Sampling interval and sampling frequency are assumed to be "
        r"approximately multiplicative inverse to each other.*",
    ):
        Signal(**inputs)
