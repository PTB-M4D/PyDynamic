from typing import Tuple

import numpy as np
import pytest
from numpy.random import default_rng
from PyDynamic import shift_uncertainty


@pytest.fixture
def rng():
    return default_rng()


@pytest.fixture
def random_standard_normal_array(rng):
    def _generate_random_array(length):
        return rng.standard_normal(length)

    return _generate_random_array


def _set_irrelevant_ranges_to_zero(
    signal: np.ndarray, uncertainties: np.ndarray, swing_in_length: int, shift: float
) -> Tuple[np.ndarray, np.ndarray]:
    relevant_signal_comparison_range_after_swing_in = np.zeros_like(signal, dtype=bool)
    relevant_uncertainty_comparison_range_after_swing_in = np.zeros_like(
        uncertainties, dtype=bool
    )
    relevant_signal_comparison_range_after_swing_in[swing_in_length:] = 1
    relevant_uncertainty_comparison_range_after_swing_in[
        swing_in_length:, swing_in_length:
    ] = 1
    (
        shifted_relevant_signal_comparison_range_after_swing_in,
        shifted_relevant_uncertainty_comparison_range_after_swing_in,
    ) = shift_uncertainty(
        relevant_signal_comparison_range_after_swing_in,
        relevant_uncertainty_comparison_range_after_swing_in,
        -int(shift),
    )
    signal[np.logical_not(shifted_relevant_signal_comparison_range_after_swing_in)] = 0
    uncertainties[
        np.logical_not(shifted_relevant_uncertainty_comparison_range_after_swing_in)
    ] = 0
    return signal, uncertainties
