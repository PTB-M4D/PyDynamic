import numpy as np
import pytest
from hypothesis import given, strategies as hst
from numpy.testing import assert_almost_equal

from PyDynamic.misc.testsignals import (
    sine,
)


@given(
    hst.floats(min_value=1, max_value=1e64, allow_infinity=False, allow_nan=False),
    hst.integers(min_value=1, max_value=1000),
)
@pytest.mark.slow
def test_freq_multiples_sine(time, hi_res_time, create_timestamps, freq, rep):
    # Create time vector with timestamps near multiples of frequency.
    fixed_freq_time = create_timestamps(time[0], rep * 1 / freq, 1 / freq)
    x = sine(fixed_freq_time, freq=freq)
    # Check if signal at multiples of frequency is start value of signal.
    for i_x in x:
        assert_almost_equal(i_x, 0)


@given(hst.floats(min_value=0, exclude_min=True, allow_infinity=False))
@pytest.mark.slow
def test_max_sine(time, amp):
    # Test if casual time signal's maximum equals the input amplitude.

    x = sine(time, amp=amp)
    # Check for minimal callability and that maximum amplitude at
    # timestamps is below default.
    assert np.max(np.abs(x)) <= amp


@given(hst.floats(min_value=0, exclude_min=True, allow_infinity=False))
@pytest.mark.slow
def test_hi_res_max_sine(hi_res_time, amp):
    # Test if high-resoluted time signal's maximum equals the input amplitude.

    # Initialize fixed amplitude.
    x = sine(hi_res_time, amp=amp)
    # Check for minimal callability with high resolution time vector and
    # that maximum amplitude at timestamps is almost equal default.
    assert_almost_equal(np.max(x), amp)
    assert_almost_equal(np.min(x), -amp)
