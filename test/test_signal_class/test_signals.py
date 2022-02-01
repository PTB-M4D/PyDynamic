import numpy as np
import pytest
from hypothesis import given, HealthCheck, settings, strategies as hst
from numpy.testing import assert_almost_equal
from pytest import approx

from PyDynamic.misc.testsignals import (
    GaussianPulse,
    multi_sine,
    rect,
    shocklikeGaussian,
    sine,
    squarepulse,
)
from PyDynamic.signals import Signal
from .conftest import signal_inputs
from ..conftest import (
    _print_during_test_to_avoid_timeout,
    hypothesis_not_negative_float,
)


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
def test_signal_class_raise_not_implemented_multivariate_signal(capsys, inputs):
    inputs["values"] = inputs["values"][..., np.newaxis]
    _print_during_test_to_avoid_timeout(capsys)
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
)
@pytest.mark.slow
def test_signal_class_raise_value_error_on_non_matching_sampling_freq_and_time_step(
    capsys, inputs
):
    inputs["Fs"] = inputs["Ts"]
    _print_during_test_to_avoid_timeout(capsys)
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
)
@pytest.mark.slow
def test_signal_class_raise_value_error_on_non_matching_dimension_of_uncertainties(
    capsys, inputs
):
    inputs["uncertainty"] = inputs["uncertainty"][:-1]
    _print_during_test_to_avoid_timeout(capsys)
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
)
@pytest.mark.slow
def test_signal_class_raise_value_error_on_non_matching_dimension_of_time_and_values(
    capsys, inputs
):
    inputs["time"] = inputs["time"][:-1]
    _print_during_test_to_avoid_timeout(capsys)
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
)
@pytest.mark.slow
def test_signal_class_raise_value_error_on_non_square_uncertainties(capsys, inputs):
    inputs["uncertainty"] = inputs["uncertainty"][..., :-1]
    _print_during_test_to_avoid_timeout(capsys)
    with pytest.raises(
        ValueError,
        match=r"Signal: if uncertainties are provided as 2-dimensional np.ndarray "
        r"they are expected to resemble a square matrix, but uncertainties are of "
        r"shape.*",
    ):
        Signal(**inputs)
