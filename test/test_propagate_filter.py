"""Perform test for uncertainty.propagate_filter"""
import itertools

import numpy as np
import pytest
import scipy

from PyDynamic.misc.testsignals import rect
from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.uncertainty.propagate_filter import (
    _get_derivative_A,
    _tf2ss,
    FIRuncFilter,
    IIRuncFilter,
)
from PyDynamic.uncertainty.propagate_MonteCarlo import MC


def random_array(length):
    array = np.random.randn(length)
    return array


def random_nonnegative_array(length):
    array = np.random.random(length)
    return array


def random_semiposdef_matrix(length):
    matrix = np.random.random((length, length))
    matrix = make_semiposdef(matrix)
    return matrix


def valid_filters():
    N = np.random.randint(2, 100)  # scipy.linalg.companion requires N >= 2
    theta = random_array(N)

    valid_filters = [
        {"theta": theta, "Utheta": None},
        {"theta": theta, "Utheta": np.zeros((N, N))},
        {"theta": theta, "Utheta": random_semiposdef_matrix(N)},
    ]

    return valid_filters


def valid_signals():
    N = np.random.randint(100, 1000)
    signal = random_array(N)

    valid_signals = [
        {"y": signal, "sigma_noise": np.random.randn(), "kind": "float"},
        {"y": signal, "sigma_noise": random_nonnegative_array(N), "kind": "diag"},
        {"y": signal, "sigma_noise": random_nonnegative_array(N // 2), "kind": "corr"},
    ]

    return valid_signals


def valid_lows():
    N = np.random.randint(2, 10)  # scipy.linalg.companion requires N >= 2
    blow = random_array(N)

    valid_lows = [
        {"blow": None},
        {"blow": blow},
    ]

    return valid_lows


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


@pytest.mark.parametrize("filters", valid_filters())
@pytest.mark.parametrize("signals", valid_signals())
@pytest.mark.parametrize("lowpasses", valid_lows())
def test_FIRuncFilter(filters, signals, lowpasses):
    # Check expected output for thinkable permutations of input parameters.
    y, Uy = FIRuncFilter(**filters, **signals, **lowpasses)
    assert len(y) == len(signals["y"])
    assert len(Uy) == len(signals["y"])

    # note: a direct comparison against scipy.signal.lfilter is not needed,
    #       as y is already computed using this method


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
        assert np.allclose(a, b)

    for a, b in itertools.combinations(all_uy, 2):
        assert np.allclose(a, b)


# in the following test, we exclude the case of a valid signal with uncertainty given as
# the right-sided auto-covariance (acf). This is done, because we currently do not ensure, that
# the random-drawn acf generates a positive-semidefinite Toeplitz-matrix. Therefore we cannot
# construct a valid and equivalent input for the Monte-Carlo method in that case.
@pytest.mark.parametrize("filters", valid_filters())
@pytest.mark.parametrize("signals", valid_signals()[:2])  # exclude kind="corr"
@pytest.mark.parametrize("lowpasses", valid_lows())
def test_FIRuncFilter_MC_uncertainty_comparison(filters, signals, lowpasses):
    # Check output for thinkable permutations of input parameters against a Monte Carlo approach.

    # run method
    y_fir, uy_fir = FIRuncFilter(**filters, **signals, **lowpasses)

    # run Monte Carlo simulation of an FIR
    ## adjust input to match conventions of MC
    x = signals["y"]
    ux = signals["sigma_noise"]

    b = filters["theta"]
    a = [1.0]
    if isinstance(filters["Utheta"], np.ndarray):
        Uab = filters["Utheta"]
    else:  # Utheta == None
        Uab = np.zeros((len(b), len(b)))  # MC-method cant deal with Utheta = None

    blow = lowpasses["blow"]
    if isinstance(blow, np.ndarray):
        n_blow = len(blow)
    else:
        n_blow = 0

    ## run FIR with MC and extract diagonal of returned covariance
    y_mc, uy_mc = MC(x, ux, b, a, Uab, blow=blow, runs=2000)
    uy_mc = np.sqrt(np.diag(uy_mc))

    # HACK: for visualization during debugging
    # import matplotlib.pyplot as plt
    # plt.plot(uy_fir, label="fir")
    # plt.plot(uy_mc, label="mc")
    # plt.title("filter: {0}, signal: {1}".format(len(b), len(x)))
    # plt.legend()
    # plt.show()
    # /HACK

    # approximative comparison after swing-in of MC-result
    # (which is after the combined length of blow and b)
    assert np.allclose(
        uy_fir[len(b) + n_blow :],
        uy_mc[len(b) + n_blow :],
        atol=1e-1,
        rtol=1e-1,
    )


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
    a = np.array([1.0,])
    Uab = np.diag(
        np.zeros((len(a) + len(b) - 1))
    )  # fully certain filter-parameters, otherwise FIR and IIR do not match! (see docstring of IIRuncFilter)

    return {"b": b, "a": a, "Uab": Uab}


@pytest.fixture(scope="module")
def input_signal():

    # simulate input and output signals
    Fs = 100e3  # sampling frequency (in Hz)
    Ts = 1 / Fs  # sampling interval length (in s)
    nx = 500
    time = np.arange(nx) * Ts  # time values

    # input signal + run methods
    sigma_noise = 1e-2  # std for input signal
    x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=sigma_noise)  # generate input signal
    Ux = sigma_noise * np.ones_like(x)  # uncertainty of input signal

    return {"x": x, "Ux": Ux}


@pytest.fixture(scope="module")
def run_IIRuncFilter_all_at_once(iir_filter, input_signal):
    y, Uy, _ = IIRuncFilter(**input_signal, **iir_filter, kind="diag" )
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
            x_batch, Ux_batch, **iir_filter, kind="diag", state=state,
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


def test_IIRuncFilter_shape_in_chunks(
    input_signal, run_IIRuncFilter_in_chunks
):
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
    assert np.allclose(y1, y2)
    assert np.allclose(Uy1, Uy2)

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

    assert np.allclose(y_fir, y_iir)
    assert np.allclose(Uy_fir, Uy_iir)


def test_tf2ss(iir_filter):
    """compare output of _tf2ss to (the very similar) scipy.signal.tf2ss"""
    b = iir_filter["b"]
    a = iir_filter["a"]
    A1, B1, C1, D1 = _tf2ss(b, a)
    A2, B2, C2, D2 = scipy.signal.tf2ss(b, a)

    assert np.allclose(A1, A2[::-1, ::-1])
    assert np.allclose(B1, B2[::-1, ::-1])
    assert np.allclose(C1, C2[::-1, ::-1])
    assert np.allclose(D1, D2[::-1, ::-1])


def test_get_derivative_A():
    """dA is sparse and only a specifc diagonal is of value -1.0"""
    p = 10
    dA = _get_derivative_A(p)
    index1 = np.arange(p)
    index2 = np.full(p, -1)
    index3 = index1[::-1]

    sliced_diagonal = np.full(p, -1.0)

    assert np.allclose(dA[index1, index2, index3], sliced_diagonal)
