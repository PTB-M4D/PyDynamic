""" Perform tests on methods to create test signals."""
from typing import Callable, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import given, HealthCheck, settings, strategies as hst
from hypothesis.strategies import composite
from numpy.testing import assert_almost_equal
from pytest import approx

from PyDynamic.examples.working_with_signals import demonstrate_signal
from PyDynamic.misc.testsignals import (
    GaussianPulse,
    multi_sine,
    rect,
    shocklikeGaussian,
    sine,
    squarepulse,
)
from PyDynamic.signals import Signal
from .conftest import (
    _print_current_ram_usage,
    hypothesis_bounded_float,
    hypothesis_covariance_matrix,
    hypothesis_float_vector,
    hypothesis_not_negative_float,
)
from .test_propagate_filter import FIRuncFilter_input


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


@composite
def apply_fir_filter_inputs(draw: Callable):
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


@pytest.fixture(scope="module")
def N():
    return 2048


@pytest.fixture(scope="module")
def Ts():
    return 0.01


@pytest.fixture(scope="module")
def hi_res_time(create_timestamps):
    return create_timestamps(0, 2 * np.pi, 1e-5)


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
def t0(N, Ts):
    return N / 2 * Ts


def test_shocklikeGaussian(Ts, t0, N, time):
    m0 = 1 + np.random.rand() * 0.2
    sigma = 50 * Ts
    # zero noise
    x = shocklikeGaussian(time, t0, m0, sigma, noise=0.0)
    assert_almost_equal(np.max(x), m0)
    assert np.std(x[: N // 10]) < 1e-10
    # noisy signal
    nstd = 1e-2
    x = shocklikeGaussian(time, t0, m0, sigma, noise=nstd)
    assert_almost_equal(np.round(np.std(x[: N // 10]) * 100) / 100, nstd)


def test_GaussianPulse(Ts, t0, N, time):
    m0 = 1 + np.random.rand() * 0.2
    sigma = 50 * Ts
    # zero noise
    x = GaussianPulse(time, t0, m0, sigma, noise=0.0)
    assert_almost_equal(np.max(x), m0)
    assert_almost_equal(time[x.argmax()], t0)
    assert np.std(x[: N // 10]) < 1e-10
    # noisy signal
    nstd = 1e-2
    x = GaussianPulse(time, t0, m0, sigma, noise=nstd)
    assert_almost_equal(np.round(np.std(x[: N // 10]) * 100) / 100, nstd)


def test_rect(Ts, t0, N, time):
    width = N // 4 * Ts
    height = 1 + np.random.rand() * 0.2
    x = rect(time, t0, t0 + width, height, noise=0.0)
    assert_almost_equal(np.max(x), height)
    assert np.max(x[time < t0]) < 1e-10
    assert np.max(x[time > t0 + width]) < 1e-10
    # noisy signal
    nstd = 1e-2
    x = rect(time, t0, t0 + width, height, noise=nstd)
    assert np.round(np.std(x[time < t0]) * 100) / 100 == approx(nstd)


def test_squarepulse(time):
    height = 1 + np.random.rand() * 0.2
    numpulses = 5
    x = squarepulse(time, height, numpulses, noise=0.0)
    assert_almost_equal(np.max(x), height)


@pytest.mark.slow
def test_minimal_call_max_sine(time):
    x = sine(time)
    # Check for minimal callability and that maximum amplitude at
    # timestamps is below default.
    assert np.max(np.abs(x)) <= 1.0


@pytest.mark.slow
def test_minimal_call_hi_res_max_sine(hi_res_time):
    x = sine(hi_res_time)
    # Check for minimal callability with high resolution time vector and
    # that maximum amplitude at timestamps is almost equal default.
    assert_almost_equal(np.max(x), 1.0)
    assert_almost_equal(np.min(x), -1.0)


@given(
    hst.floats(min_value=1, max_value=1e64, allow_infinity=False, allow_nan=False),
    hst.integers(min_value=1, max_value=1000),
)
@pytest.mark.slow
def test_medium_call_freq_multiples_sine(
    time, hi_res_time, create_timestamps, freq, rep
):
    # Create time vector with timestamps near multiples of frequency.
    fixed_freq_time = create_timestamps(time[0], rep * 1 / freq, 1 / freq)
    x = sine(fixed_freq_time, freq=freq)
    # Check if signal at multiples of frequency is start value of signal.
    for i_x in x:
        assert_almost_equal(i_x, 0)


@given(hst.floats(min_value=0, exclude_min=True, allow_infinity=False))
@pytest.mark.slow
def test_medium_call_max_sine(time, amp):
    # Test if casual time signal's maximum equals the input amplitude.

    x = sine(time, amp=amp)
    # Check for minimal callability and that maximum amplitude at
    # timestamps is below default.
    assert np.max(np.abs(x)) <= amp


@given(hst.floats(min_value=0, exclude_min=True, allow_infinity=False))
@pytest.mark.slow
def test_medium_call_hi_res_max_sine(hi_res_time, amp):
    # Test if high-resoluted time signal's maximum equals the input amplitude.

    # Initialize fixed amplitude.
    x = sine(hi_res_time, amp=amp)
    # Check for minimal callability with high resolution time vector and
    # that maximum amplitude at timestamps is almost equal default.
    assert_almost_equal(np.max(x), amp)
    assert_almost_equal(np.min(x), -amp)


@given(hst.floats(), hst.floats(), hypothesis_not_negative_float(allow_infinity=True))
def test_full_call_sine(time, amp, freq, noise):
    # Check for all possible calls.
    sine(time)
    sine(time, amp)
    sine(time, amp=amp)
    sine(time, amp, freq)
    sine(time, amp, freq=freq)
    sine(time, amp=amp, freq=freq)
    sine(time, freq=freq)
    sine(time, amp, freq, noise=noise)
    sine(time, noise=noise)
    sine(time, amp, freq=freq, noise=noise)
    sine(time, freq=freq, noise=noise)
    sine(time, amp=amp, freq=freq, noise=noise)


@given(hst.floats(), hst.floats())
def test_compare_multisine_with_sine(time, freq, amp):
    # Compare the result of a call of sine and a similar call of multi_sine
    # with one-element lists of amplitudes and frequencies.

    x = sine(time=time, amp=amp, freq=freq)
    multi_x = multi_sine(time=time, amps=[amp], freqs=[freq])
    # Check for minimal callability and that maximum amplitude at
    # timestamps is below default.
    assert_almost_equal(x, multi_x)


@pytest.mark.slow
def test_signal_example(monkeypatch):
    # Test executability of the demonstrate_signal example.
    # With this expression we override the matplotlib.pyplot.show method with a
    # lambda expression returning None but only for this one test.
    monkeypatch.setattr(plt, "show", lambda: None, raising=True)
    demonstrate_signal()


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
    _print_current_ram_usage(capsys)
    Signal(**inputs)


@given(signal_inputs())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
    max_examples=10,
)
@pytest.mark.slow
def test_signal_class_raise_not_implemented(capsys, inputs):
    inputs["values"] = inputs["values"][..., np.newaxis]
    _print_current_ram_usage(capsys)
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
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
    max_examples=10,
)
@pytest.mark.slow
def test_signal_class_raise_value_error_on_non_matching_sampling_freq_and_time_step(
    capsys, inputs
):
    inputs["Fs"] = inputs["Ts"]
    _print_current_ram_usage(capsys)
    with pytest.raises(
        ValueError,
        match=r"Signal: Sampling interval and sampling frequency are assumed to be "
        r"approximately multiplicative inverse to each other.*",
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
    max_examples=10,
)
@pytest.mark.slow
def test_signal_class_raise_value_error_on_non_matching_dimension_of_uncertainties(
    capsys, inputs
):
    inputs["uncertainty"] = inputs["uncertainty"][:-1]
    _print_current_ram_usage(capsys)
    with pytest.raises(
        ValueError,
        match=r"Signal: if uncertainties are provided as np.ndarray "
        r"they are expected to match the number of elements of the "
        r"provided time vector, but uncertainties are of shape.*",
    ):
        Signal(**inputs)


@given(signal_inputs())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
    max_examples=10,
)
@pytest.mark.slow
def test_signal_class_raise_value_error_on_non_matching_dimension_of_time_and_values(
    capsys, inputs
):
    inputs["time"] = inputs["time"][:-1]
    _print_current_ram_usage(capsys)
    with pytest.raises(
        ValueError,
        match=r"Signal: Number of elements of the provided time and signal vectors "
        "are expected to match, but time is of length.*",
    ):
        Signal(**inputs)


@given(signal_inputs(ensure_uncertainty_covariance_matrix=True))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
    max_examples=10,
)
@pytest.mark.slow
def test_signal_class_raise_value_error_on_non_square_uncertainties(capsys, inputs):
    inputs["uncertainty"] = inputs["uncertainty"][..., :-1]
    _print_current_ram_usage(capsys)
    with pytest.raises(
        ValueError,
        match=r"Signal: if uncertainties are provided as 2-dimensional np.ndarray "
        r"they are expected to resemble a square matrix, but uncertainties are of "
        r"shape.*",
    ):
        Signal(**inputs)


@given(apply_fir_filter_inputs())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
    max_examples=10,
)
@pytest.mark.slow
def test_signal_apply_fir_filter(capsys, signal_and_filter_inputs):
    _print_current_ram_usage(capsys)
    signal_init_inputs, filter_inputs = signal_and_filter_inputs
    test_signal = Signal(**signal_init_inputs)
    test_signal.apply_filter(**filter_inputs)
