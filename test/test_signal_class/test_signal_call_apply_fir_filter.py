from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pytest
from hypothesis import given, HealthCheck, settings
from hypothesis.strategies import composite

from PyDynamic.signals import Signal
from .conftest import signal_inputs
from ..conftest import (
    _print_during_test_to_avoid_timeout,
    FIRuncFilter_input,
)


@composite
def apply_fir_filter_inputs(
    draw: Callable,
) -> Tuple[
    Dict[str, Optional[Union[float, np.ndarray]]],
    Dict[str, Optional[Union[float, np.ndarray]]],
]:
    filter_inputs = draw(FIRuncFilter_input(exclude_corr_kind=True))
    signal_init_inputs = draw(
        signal_inputs(force_number_of_samples_to=len(filter_inputs["y"]))
    )
    signal_init_inputs["values"] = filter_inputs["y"]
    signal_init_inputs["uncertainty"] = filter_inputs["sigma_noise"]
    signals_apply_filter_inputs = {
        "b": filter_inputs["theta"],
        "filter_uncertainty": filter_inputs["Utheta"],
    }
    return signal_init_inputs, signals_apply_filter_inputs


@given(apply_fir_filter_inputs())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)
@pytest.mark.slow
def test(capsys, signal_and_filter_inputs):
    _print_during_test_to_avoid_timeout(capsys)
    signal_init_inputs, filter_inputs = signal_and_filter_inputs
    test_signal = Signal(**signal_init_inputs)
    test_signal.apply_filter(**filter_inputs)
    assert np.all(test_signal.values)
    assert np.all(test_signal.uncertainty)
