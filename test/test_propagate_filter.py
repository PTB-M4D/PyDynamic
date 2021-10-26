"""Perform test for uncertainty.propagate_filter"""
import itertools
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest
import scipy
from hypothesis import given, HealthCheck, settings, strategies as hst
from hypothesis.strategies import composite
from numpy.testing import assert_allclose, assert_equal
from scipy.linalg import toeplitz
from scipy.signal import lfilter, lfilter_zi

from PyDynamic.misc.testsignals import rect
from PyDynamic.misc.tools import shift_uncertainty, trimOrPad

# noinspection PyProtectedMember
from PyDynamic.uncertainty.propagate_filter import (
    _fir_filter,
    _get_derivative_A,
    _tf2ss,
    FIRuncFilter,
    IIRuncFilter,
)
from PyDynamic.uncertainty.propagate_MonteCarlo import MC
from .conftest import (
    hypothesis_covariance_matrix,
    hypothesis_float_vector,
    hypothesis_not_negative_float,
    random_covariance_matrix,
    scale_matrix_or_vector_to_range,
)


@composite
def FIRuncFilter_input(
    draw: Callable, exclude_corr_kind: bool = False
) -> Dict[str, Any]:
    filter_length = draw(
        hst.integers(min_value=2, max_value=100)
    )  # scipy.linalg.companion requires N >= 2
    filter_theta = draw(
        hypothesis_float_vector(length=filter_length, min_value=1e-2, max_value=1e3)
    )
    filter_theta_covariance = draw(
        hst.one_of(
            hypothesis_covariance_matrix(number_of_rows=filter_length), hst.just(None)
        )
    )

    signal_length = draw(hst.integers(min_value=200, max_value=1000))
    signal = draw(
        hypothesis_float_vector(length=signal_length, min_value=1e-2, max_value=1e3)
    )

    if exclude_corr_kind:
        kind = draw(hst.sampled_from(("float", "diag")))
    else:
        kind = draw(hst.sampled_from(("float", "diag", "corr")))
    if kind == "diag":
        signal_standard_deviation = draw(
            hypothesis_float_vector(length=signal_length, min_value=0, max_value=1e2)
        )
    elif kind == "corr":
        random_data = draw(
            hypothesis_float_vector(
                length=signal_length // 2, min_value=1e-2, max_value=1e2
            )
        )
        acf = scipy.signal.correlate(random_data, random_data, mode="full")

        scaled_acf = scale_matrix_or_vector_to_range(
            acf, range_min=1e-10, range_max=1e3
        )
        signal_standard_deviation = scaled_acf[len(scaled_acf) // 2 :]
    else:
        signal_standard_deviation = draw(hypothesis_not_negative_float(max_value=1e2))

    lowpass_length = draw(
        hst.integers(min_value=2, max_value=10)
    )  # scipy.linalg.companion requires N >= 2
    lowpass = draw(
        hst.one_of(
            (
                hypothesis_float_vector(
                    length=lowpass_length, min_value=1e-2, max_value=1e2
                ),
                hst.just(None),
            )
        )
    )

    shift = draw(hypothesis_not_negative_float(max_value=1e2))

    return {
        "theta": filter_theta,
        "Utheta": filter_theta_covariance,
        "y": signal,
        "sigma_noise": signal_standard_deviation,
        "kind": kind,
        "blow": lowpass,
        "shift": shift,
    }


def random_array(length):
    return np.random.randn(length)


@pytest.fixture
def equal_filters():
    # Create two filters with assumed identical FIRuncFilter() output to test
    # equality of the more efficient with the standard implementation.

    N = np.random.randint(2, 100)  # scipy.linalg.companion requires N >= 2
    theta = random_array(N)

    equal_filters = [
        {"theta": theta, "Utheta": None},
        {"theta": theta, "Utheta": np.zeros((N, N))},
    ]

    return equal_filters


@pytest.fixture
def equal_signals():
    # Create three signals with assumed identical FIRuncFilter() output to test
    # equality of the different cases of input parameter 'kind'.

    # some shortcuts
    N = np.random.randint(100, 1000)
    signal = random_array(N)
    s = np.random.randn()
    acf = np.array([s ** 2] + [0] * (N - 1))

    equal_signals = [
        {"y": signal, "sigma_noise": s, "kind": "float"},
        {"y": signal, "sigma_noise": np.full(N, s), "kind": "diag"},
        {"y": signal, "sigma_noise": acf, "kind": "corr"},
    ]

    return equal_signals


@given(FIRuncFilter_input())
@settings(deadline=None)
@pytest.mark.slow
def test_FIRuncFilter(fir_unc_filter_input):
    # Check expected output for thinkable permutations of input parameters.
    y, Uy = FIRuncFilter(**fir_unc_filter_input)
    assert len(y) == len(fir_unc_filter_input["y"])
    assert len(Uy) == len(fir_unc_filter_input["y"])

    # note: a direct comparison against scipy.signal.lfilter is not needed,
    #       as y is already computed using this method


def legacy_FIRuncFilter(
    y, sigma_noise, theta, Utheta=None, shift=0, blow=None, kind="corr"
):
    """Uncertainty propagation for signal y and uncertain FIR filter theta

    A preceding FIR low-pass filter with coefficients `blow` can be provided optionally.

    Parameters
    ----------
    y : np.ndarray
        filter input signal
    sigma_noise : float or np.ndarray
        float: standard deviation of white noise in y
        1D-array: interpretation depends on kind
    theta : np.ndarray
        FIR filter coefficients
    Utheta : np.ndarray, optional
        covariance matrix associated with theta
        if the filter is fully certain, use `Utheta = None` (default) to make use of
        more efficient calculations. See also the comparison given in
        <examples/Digital filtering/FIRuncFilter_runtime_comparison.py>
    shift : int, optional
        time delay of filter output signal (in samples) (defaults to 0)
    blow : np.ndarray, optional
        optional FIR low-pass filter
    kind : string
        only meaningful in combination with sigma_noise a 1D numpy array
        "diag": point-wise standard uncertainties of non-stationary white noise
        "corr": single sided autocovariance of stationary (colored/correlated)
        noise (default)

    Returns
    -------
    x : np.ndarray
        FIR filter output signal
    ux : np.ndarray
        point-wise standard uncertainties associated with x


    References
    ----------

    * Elster and Link 2008 [Elster2008]_

    .. seealso:: :mod:`PyDynamic.deconvolution.fit_filter`

    """

    Ntheta = len(theta)  # FIR filter size

    # check which case of sigma_noise is necessary
    if isinstance(sigma_noise, float):
        sigma2 = sigma_noise ** 2

    elif isinstance(sigma_noise, np.ndarray) and len(sigma_noise.shape) == 1:
        if kind == "diag":
            sigma2 = sigma_noise ** 2
        elif kind == "corr":
            sigma2 = sigma_noise
        else:
            raise ValueError("unknown kind of sigma_noise")

    else:
        raise ValueError(
            f"FIRuncFilter: Uncertainty sigma_noise associated "
            f"with input signal is expected to be either a float or a 1D array but "
            f"is of shape {sigma_noise.shape}. Please check documentation for input "
            f"parameters sigma_noise and kind for more information."
        )

    if isinstance(
        blow, np.ndarray
    ):  # calculate low-pass filtered signal and propagate noise

        if isinstance(sigma2, float):
            Bcorr = np.correlate(blow, blow, "full")  # len(Bcorr) == 2*Ntheta - 1
            ycorr = (
                sigma2 * Bcorr[len(blow) - 1 :]
            )  # only the upper half of the correlation is needed

            # trim / pad to length Ntheta
            ycorr = trimOrPad(ycorr, Ntheta)
            Ulow = toeplitz(ycorr)

        elif isinstance(sigma2, np.ndarray):

            if kind == "diag":
                # [Leeuw1994](Covariance matrix of ARMA errors in closed form) can be
                # used, to derive this formula
                # The given "blow" corresponds to a MA(q)-process.
                # Going through the calculations of Leeuw, but assuming
                # that E(vv^T) is a diagonal matrix with non-identical elements,
                # the covariance matrix V becomes (see Leeuw:corollary1)
                # V = N * SP * N^T + M * S * M^T
                # N, M are defined as in the paper
                # and SP is the covariance of input-noise prior to the observed
                # time-interval (SP needs be available len(blow)-time steps into the
                # past. Here it is assumed, that SP is constant with the first value
                # of sigma2)

                # V needs to be extended to cover Ntheta-1 time steps more into the past
                sigma2_extended = np.append(sigma2[0] * np.ones((Ntheta - 1)), sigma2)

                N = toeplitz(blow[1:][::-1], np.zeros_like(sigma2_extended)).T
                M = toeplitz(
                    trimOrPad(blow, len(sigma2_extended)),
                    np.zeros_like(sigma2_extended),
                )
                SP = np.diag(sigma2[0] * np.ones_like(blow[1:]))
                S = np.diag(sigma2_extended)

                # Ulow is to be sliced from V, see below
                V = N.dot(SP).dot(N.T) + M.dot(S).dot(M.T)

            elif kind == "corr":

                # adjust the lengths sigma2 to fit blow and theta
                # this either crops (unused) information or appends zero-information
                # note1: this is the reason, why Ulow will have dimension
                # (Ntheta x Ntheta) without further ado

                # calculate Bcorr
                Bcorr = np.correlate(blow, blow, "full")

                # pad or crop length of sigma2, then reflect some part to the left and
                # invert the order
                # [0 1 2 3 4 5 6 7] --> [0 0 0 7 6 5 4 3 2 1 0 1 2 3]
                sigma2 = trimOrPad(sigma2, len(blow) + Ntheta - 1)
                sigma2_reflect = np.pad(sigma2, (len(blow) - 1, 0), mode="reflect")

                ycorr = np.correlate(
                    sigma2_reflect, Bcorr, mode="valid"
                )  # used convolve in a earlier version, should make no difference as
                # Bcorr is symmetric
                Ulow = toeplitz(ycorr)

        xlow, _ = lfilter(blow, 1.0, y, zi=y[0] * lfilter_zi(blow, 1.0))

    else:  # if blow is not provided
        if isinstance(sigma2, float):
            Ulow = np.eye(Ntheta) * sigma2

        elif isinstance(sigma2, np.ndarray):

            if kind == "diag":
                # V needs to be extended to cover Ntheta time steps more into the past
                sigma2_extended = np.append(sigma2[0] * np.ones((Ntheta - 1)), sigma2)

                # Ulow is to be sliced from V, see below
                V = np.diag(
                    sigma2_extended
                )  # this is not Ulow, same thing as in the case of a provided blow
                # (see above)

            elif kind == "corr":
                Ulow = toeplitz(trimOrPad(sigma2, Ntheta))

        xlow = y

    # apply FIR filter to calculate best estimate in accordance with GUM
    x, _ = lfilter(theta, 1.0, xlow, zi=xlow[0] * lfilter_zi(theta, 1.0))
    x = np.roll(x, -int(shift))

    # add dimension to theta, otherwise transpose won't work
    if len(theta.shape) == 1:
        theta = theta[:, np.newaxis]

    # NOTE: In the code below wherever `theta` or `Utheta` get used, they need to be
    # flipped. This is necessary to take the time-order of both variables into
    # account. (Which is descending for `theta` and `Utheta` but ascending for `Ulow`.)
    #
    # Further details and illustrations showing the effect of not-flipping
    # can be found at https://github.com/PTB-M4D/PyDynamic/issues/183

    # handle diag-case, where Ulow needs to be sliced from V
    if kind == "diag":
        # UncCov needs to be calculated inside in its own for-loop
        # V has dimension (len(sigma2) + Ntheta) * (len(sigma2) + Ntheta) --> slice a
        # fitting Ulow of dimension (Ntheta x Ntheta)
        UncCov = np.zeros((len(sigma2)))

        if isinstance(Utheta, np.ndarray):
            for k in range(len(sigma2)):
                Ulow = V[k : k + Ntheta, k : k + Ntheta]
                UncCov[k] = np.squeeze(
                    np.flip(theta).T.dot(Ulow.dot(np.flip(theta)))
                    + np.abs(np.trace(Ulow.dot(np.flip(Utheta))))
                )  # static part of uncertainty
        else:
            for k in range(len(sigma2)):
                Ulow = V[k : k + Ntheta, k : k + Ntheta]
                UncCov[k] = np.squeeze(
                    np.flip(theta).T.dot(Ulow.dot(np.flip(theta)))
                )  # static part of uncertainty

    else:
        if isinstance(Utheta, np.ndarray):
            UncCov = np.flip(theta).T.dot(Ulow.dot(np.flip(theta))) + np.abs(
                np.trace(Ulow.dot(np.flip(Utheta)))
            )  # static part of uncertainty
        else:
            UncCov = np.flip(theta).T.dot(
                Ulow.dot(np.flip(theta))
            )  # static part of uncertainty

    if isinstance(Utheta, np.ndarray):
        unc = np.empty_like(y)

        # use extended signal to match assumption of stationary signal prior to first
        # entry
        xlow_extended = np.append(np.full(Ntheta - 1, xlow[0]), xlow)

        for m in range(len(xlow)):
            # extract necessary part from input signal
            XL = xlow_extended[m : m + Ntheta, np.newaxis]
            unc[m] = XL.T.dot(np.flip(Utheta).dot(XL))  # apply formula from paper
    else:
        unc = np.zeros_like(y)

    ux = np.sqrt(np.abs(UncCov + unc))
    ux = np.roll(ux, -int(shift))  # correct for delay

    return x, ux.flatten()  # flatten in case that we still have 2D array


def test_FIRuncFilter_equality(equal_filters, equal_signals):
    # Check expected output for being identical across different equivalent input
    # parameter cases.
    all_y = []
    all_uy = []

    # run all combinations of filter and signals
    for (f, s) in itertools.product(equal_filters, equal_signals):
        y, uy = FIRuncFilter(**f, **s)
        all_y.append(y)
        all_uy.append(uy)

    # check that all have the same output, as they are supposed to represent equal cases
    for a, b in itertools.combinations(all_y, 2):
        assert_allclose(a, b)

    for a, b in itertools.combinations(all_uy, 2):
        assert_allclose(a, b)


@given(FIRuncFilter_input())
@settings(deadline=None)
@pytest.mark.slow
def test_FIRuncFilter_for_correct_output_dimensions_for_full_covariance(
    fir_unc_filter_input,
):
    y_fir, Uy_fir = FIRuncFilter(**fir_unc_filter_input, return_full_covariance=True)
    assert_equal(len(fir_unc_filter_input["y"]), len(y_fir))
    assert_equal(Uy_fir.shape, (len(y_fir), len(y_fir)))


@given(FIRuncFilter_input())
@settings(deadline=None)
@pytest.mark.slow
def test_FIRuncFilter_for_correct_output_dimensions_for_vector_covariance(
    fir_unc_filter_input,
):
    _, Uy = FIRuncFilter(**fir_unc_filter_input)
    assert_equal(Uy.shape, (len(fir_unc_filter_input["y"]),))


@given(FIRuncFilter_input())
@settings(deadline=None)
@pytest.mark.slow
def test_FIRuncFilter_for_correct_dimension_of_y(fir_unc_filter_input):
    y_fir = FIRuncFilter(**fir_unc_filter_input)[0]
    assert_equal(len(fir_unc_filter_input["y"]), len(y_fir))


# in the following test, we exclude the case of a valid signal with uncertainty
# given as
# the right-sided auto-covariance (acf). This is done, because we currently do not
# ensure, that the random-drawn acf generates a positive-semidefinite
# Toeplitz-matrix. Therefore we cannot construct a valid and equivalent input for
# the
# Monte-Carlo method in that case.
@given(FIRuncFilter_input(exclude_corr_kind=True))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)
@pytest.mark.slow
def test_FIRuncFilter_MC_uncertainty_comparison(capsys, fir_unc_filter_input):
    # Check output for thinkable permutations of input parameters against a Monte Carlo
    # approach.

    # run method
    y_fir, Uy_fir = FIRuncFilter(**fir_unc_filter_input, return_full_covariance=True)

    # run Monte Carlo simulation of an FIR
    # adjust input to match conventions of MC
    x = fir_unc_filter_input["y"]
    ux = fir_unc_filter_input["sigma_noise"]

    b = fir_unc_filter_input["theta"]
    a = np.ones(1)
    if isinstance(fir_unc_filter_input["Utheta"], np.ndarray):
        Uab = fir_unc_filter_input["Utheta"]
    else:  # Utheta == None
        Uab = np.zeros((len(b), len(b)))  # MC-method cant deal with Utheta = None

    blow = fir_unc_filter_input["blow"]
    if isinstance(blow, np.ndarray):
        n_blow = len(blow)
    else:
        n_blow = 0

    # run FIR with MC and extract diagonal of returned covariance
    with capsys.disabled():
        y_mc, Uy_mc = MC(
            x,
            ux,
            b,
            a,
            Uab,
            blow=blow,
            runs=2000,
            shift=-fir_unc_filter_input["shift"],
            verbose=True,
        )

    # approximate comparison after swing-in of MC-result (which is after the combined
    # length of blow and b)
    swing_in_length = len(b) + n_blow
    relevant_y_fir, relevant_Uy_fir = _set_irrelevant_ranges_to_zero(
        signal=y_fir,
        uncertainties=Uy_fir,
        swing_in_length=swing_in_length,
        shift=fir_unc_filter_input["shift"],
    )
    relevant_y_mc, relevant_Uy_mc = _set_irrelevant_ranges_to_zero(
        signal=y_mc,
        uncertainties=Uy_mc,
        swing_in_length=swing_in_length,
        shift=fir_unc_filter_input["shift"],
    )

    # HACK for visualization during debugging
    # from PyDynamic.misc.tools import plot_vectors_and_covariances_comparison
    #
    # plot_vectors_and_covariances_comparison(
    #     vector_1=relevant_y_fir,
    #     vector_2=relevant_y_mc,
    #     covariance_1=relevant_Uy_fir,
    #     covariance_2=relevant_Uy_mc,
    #     label_1="fir",
    #     label_2="mc",
    #     title=f"filter length: {len(b)}, signal length: {len(x)}, blow: "
    #     f"{fir_unc_filter_input['blow']}",
    # )
    # /HACK
    assert_allclose(
        relevant_y_fir,
        relevant_y_mc,
        atol=np.max((np.max(np.abs(y_fir)), 1e-1)),
    )
    assert_allclose(
        relevant_Uy_fir,
        relevant_Uy_mc,
        atol=np.max((np.max(Uy_fir), 1e-7)),
    )


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


@given(FIRuncFilter_input())
@settings(deadline=None)
@pytest.mark.slow
def test_FIRuncFilter_non_negative_main_diagonal_covariance(fir_unc_filter_input):
    _, Uy_fir = FIRuncFilter(**fir_unc_filter_input, return_full_covariance=True)
    assert np.all(np.diag(Uy_fir) >= 0)


@given(FIRuncFilter_input())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
    ],
)
@pytest.mark.slow
def test_FIRuncFilter_legacy_comparison(capsys, fir_unc_filter_input):
    legacy_y, legacy_Uy = legacy_FIRuncFilter(**fir_unc_filter_input)
    with capsys.disabled():
        current_y, current_Uy = FIRuncFilter(**fir_unc_filter_input)

    assert_allclose(
        legacy_y,
        current_y,
        atol=1e-15,
    )
    assert_allclose(
        legacy_Uy,
        current_Uy,
        atol=np.max((np.max(current_Uy) * 1e-7, 1e-7)),
        rtol=3e-6,
    )


@pytest.mark.slow
def test_fir_filter_MC_comparison():
    N_signal = np.random.randint(20, 25)
    x = random_array(N_signal)
    Ux = random_covariance_matrix(N_signal)

    N_theta = np.random.randint(2, 5)  # scipy.linalg.companion requires N >= 2
    theta = random_array(N_theta)  # scipy.signal.firwin(N_theta, 0.1)
    Utheta = random_covariance_matrix(N_theta)

    # run method
    y_fir, Uy_fir = _fir_filter(x, theta, Ux, Utheta, initial_conditions="zero")

    # run FIR with MC and extract diagonal of returned covariance
    y_mc, Uy_mc = MC(x, Ux, theta, np.ones(1), Utheta, blow=None, runs=10000)

    # HACK: for visualization during debugging
    # from PyDynamic.misc.tools import plot_vectors_and_covariances_comparison
    # plot_vectors_and_covariances_comparison(
    #     vector_1=y_fir,
    #     vector_2=y_mc,
    #     covariance_1=Uy_fir,
    #     covariance_2=Uy_mc,
    #     label_1="fir",
    #     label_2="mc",
    #     title=f"filter: {len(theta)}, signal: {len(x)}",
    # )
    # /HACK
    assert_allclose(y_fir, y_mc, atol=1e-1, rtol=1e-1)
    assert_allclose(Uy_fir, Uy_mc, atol=1e-1, rtol=1e-1)


def test_IIRuncFilter():
    pass


@pytest.fixture(scope="module")
def iir_filter():
    b = np.array([0.01967691, -0.01714282, 0.03329653, -0.01714282, 0.01967691])
    a = np.array([1.0, -3.03302405, 3.81183153, -2.29112937, 0.5553678])
    Uab = np.diag(np.zeros((len(a) + len(b) - 1)))  # uncertainty of IIR-parameters
    Uab[2, 2] = 0.000001  # only a2 is uncertain

    return {"b": b, "a": a, "Uab": Uab}


@pytest.fixture(scope="module")
def fir_filter():
    b = np.array([0.01967691, -0.01714282, 0.03329653, -0.01714282, 0.01967691])
    a = np.array(
        [
            1.0,
        ]
    )
    Uab = np.diag(
        np.zeros((len(a) + len(b) - 1))
    )  # fully certain filter-parameters, otherwise FIR and IIR do not match! (see
    # docstring of IIRuncFilter)

    return {"b": b, "a": a, "Uab": Uab}


@pytest.fixture(scope="module")
def sigma_noise():
    return 1e-2  # std for input signal


@pytest.fixture(scope="module")
def input_signal(sigma_noise):

    # simulate input and output signals
    Fs = 100e3  # sampling frequency (in Hz)
    Ts = 1 / Fs  # sampling interval length (in s)
    nx = 500
    time = np.arange(nx) * Ts  # time values

    # input signal + run methods
    x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=sigma_noise)  # generate input signal
    Ux = sigma_noise * np.ones_like(x)  # uncertainty of input signal

    return {"x": x, "Ux": Ux}


@pytest.fixture(scope="module")
def run_IIRuncFilter_all_at_once(iir_filter, input_signal):
    y, Uy, _ = IIRuncFilter(**input_signal, **iir_filter, kind="diag")
    return y, Uy


@pytest.fixture(scope="module")
def run_IIRuncFilter_in_chunks(iir_filter, input_signal):

    # slice x into smaller chunks and process them in batches
    # this tests the internal state options
    y_list = []
    Uy_list = []
    state = None
    for x_batch, Ux_batch in zip(
        np.array_split(input_signal["x"], 200), np.array_split(input_signal["Ux"], 200)
    ):
        yi, Uyi, state = IIRuncFilter(
            x_batch,
            Ux_batch,
            **iir_filter,
            kind="diag",
            state=state,
        )
        y_list.append(yi)
        Uy_list.append(Uyi)
    y = np.concatenate(y_list, axis=0)
    Uy = np.concatenate(Uy_list, axis=0)

    return y, Uy


def test_IIRuncFilter_shape_all_at_once(input_signal, run_IIRuncFilter_all_at_once):

    y, Uy = run_IIRuncFilter_all_at_once

    # compare lengths
    assert len(y) == len(input_signal["x"])
    assert len(Uy) == len(input_signal["Ux"])


def test_IIRuncFilter_shape_in_chunks(input_signal, run_IIRuncFilter_in_chunks):
    y, Uy = run_IIRuncFilter_in_chunks

    # compare lengths
    assert len(y) == len(input_signal["x"])
    assert len(Uy) == len(input_signal["Ux"])


def test_IIRuncFilter_identity_nonchunk_chunk(
    run_IIRuncFilter_all_at_once, run_IIRuncFilter_in_chunks
):

    y1, Uy1 = run_IIRuncFilter_all_at_once
    y2, Uy2 = run_IIRuncFilter_in_chunks

    # check if both ways of calling IIRuncFilter yield the same result
    assert_allclose(y1, y2)
    assert_allclose(Uy1, Uy2)


@pytest.mark.parametrize("kind", ["diag", "corr"])
def test_FIR_IIR_identity(kind, fir_filter, input_signal):

    # run signal through both implementations
    y_iir, Uy_iir, _ = IIRuncFilter(*input_signal.values(), **fir_filter, kind=kind)
    y_fir, Uy_fir = FIRuncFilter(
        *input_signal.values(),
        theta=fir_filter["b"],
        Utheta=fir_filter["Uab"],
        kind=kind,
    )

    assert_allclose(y_fir, y_iir)
    assert_allclose(Uy_fir, Uy_iir)


def test_tf2ss(iir_filter):
    """compare output of _tf2ss to (the very similar) scipy.signal.tf2ss"""
    b = iir_filter["b"]
    a = iir_filter["a"]
    A1, B1, C1, D1 = _tf2ss(b, a)
    A2, B2, C2, D2 = scipy.signal.tf2ss(b, a)

    assert_allclose(A1, A2[::-1, ::-1])
    assert_allclose(B1, B2[::-1, ::-1])
    assert_allclose(C1, C2[::-1, ::-1])
    assert_allclose(D1, D2[::-1, ::-1])


def test_get_derivative_A():
    """dA is sparse and only a specific diagonal is of value -1.0"""
    p = 10
    dA = _get_derivative_A(p)
    index1 = np.arange(p)
    index2 = np.full(p, -1)
    index3 = index1[::-1]

    sliced_diagonal = np.full(p, -1.0)

    assert_allclose(dA[index1, index2, index3], sliced_diagonal)


def test_IIRuncFilter_raises_warning_for_kind_not_diag_with_scalar_covariance(
    sigma_noise, iir_filter, input_signal
):
    input_signal["Ux"] = sigma_noise
    with pytest.warns(UserWarning):
        IIRuncFilter(**input_signal, **iir_filter, kind="corr")
