from typing import Callable, Dict, Union

import numpy as np
import pytest
from hypothesis import strategies as hst
from hypothesis.strategies import composite

from PyDynamic.misc.testsignals import (
    rect,
)
from ..conftest import (
    hypothesis_bounded_float,
    hypothesis_covariance_matrix,
    hypothesis_float_vector,
)


@composite
def signal_inputs(
    draw: Callable,
    force_number_of_samples_to: int = None,
    ensure_time_step_to_be_float: bool = False,
    ensure_uncertainty_array: bool = False,
    ensure_uncertainty_covariance_matrix: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    minimum_float = 1e-5
    maximum_float = 1e-1
    number_of_samples = force_number_of_samples_to or draw(
        hst.integers(min_value=4, max_value=512)
    )
    small_positive_float_strategy = hypothesis_bounded_float(
        min_value=minimum_float, max_value=maximum_float
    )
    freq_strategy = hst.one_of(
        small_positive_float_strategy,
        hst.just(None),
    )
    intermediate_time_step = draw(small_positive_float_strategy)
    max_time = number_of_samples * intermediate_time_step
    time = np.arange(0, max_time, intermediate_time_step)[:number_of_samples]
    if ensure_time_step_to_be_float:
        time_step = intermediate_time_step
    else:
        time_step = draw(hst.sampled_from((intermediate_time_step, None)))
    if time_step is None:
        sampling_frequency = draw(freq_strategy)
    else:
        sampling_frequency = draw(hst.sampled_from((np.reciprocal(time_step), None)))
    values = rect(time, max_time // 4, max_time // 4 * 3)
    uncertainties_covariance_strategy = (
        hypothesis_covariance_matrix(number_of_rows=number_of_samples),
    )
    uncertainties_array_strategies = uncertainties_covariance_strategy + (
        hypothesis_float_vector(
            min_value=minimum_float, max_value=maximum_float, length=number_of_samples
        ),
    )
    if ensure_uncertainty_covariance_matrix:
        uncertainties_strategies = uncertainties_covariance_strategy
    elif ensure_uncertainty_array:
        uncertainties_strategies = uncertainties_array_strategies
    else:
        uncertainties_strategies = uncertainties_array_strategies + (
            small_positive_float_strategy,
            hst.just(None),
        )

    ux = draw(hst.one_of(uncertainties_strategies))
    return {
        "time": time,
        "values": values,
        "Ts": time_step,
        "Fs": sampling_frequency,
        "uncertainty": ux,
    }


@pytest.fixture(scope="module")
def N():
    return 2048


@pytest.fixture(scope="module")
def Ts():
    return 0.01


@pytest.fixture(scope="module")
def create_timestamps(N, Ts):
    def timestamps(t_0=0, t_n=N * Ts, d_t=Ts):
        # Compute equally spaced timestamps between t_0 and t_n.
        return np.arange(t_0, t_n, d_t)

    return timestamps


@pytest.fixture(scope="module")
def time(create_timestamps, N, Ts):
    return create_timestamps(0, N * Ts, Ts)


@pytest.fixture(scope="module")
def hi_res_time(create_timestamps):
    return create_timestamps(0, 2 * np.pi, 1e-5)
