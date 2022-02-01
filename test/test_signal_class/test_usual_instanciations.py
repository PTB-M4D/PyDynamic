import numpy as np
import pytest
from hypothesis import given, HealthCheck, settings
from numpy.testing import assert_equal

from PyDynamic.signals import Signal
from .conftest import signal_inputs
from ..conftest import (
    _print_during_test_to_avoid_timeout,
)


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
def test_signal_class_usual_instanciations(capsys, inputs):
    _print_during_test_to_avoid_timeout(capsys)
    test_signal = Signal(**inputs)
    assert test_signal.Ts is not None
    assert test_signal.Fs is not None
    assert isinstance(test_signal.uncertainty, np.ndarray)
    assert isinstance(test_signal.standard_uncertainties, np.ndarray)
    assert_equal(len(test_signal.uncertainty), len(test_signal.standard_uncertainties))
    assert_equal(len(test_signal.standard_uncertainties), len(test_signal.time))
    assert test_signal.name
    assert isinstance(test_signal.name, str)
    assert test_signal.unit_time
    assert isinstance(test_signal.unit_time, str)
    assert test_signal.unit_values
    assert isinstance(test_signal.unit_values, str)
