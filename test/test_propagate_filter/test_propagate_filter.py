import itertools

import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose
from scipy.signal import lfilter

from PyDynamic.misc.testsignals import rect

# noinspection PyProtectedMember
from PyDynamic.uncertainty.propagate_filter import (
    _fir_filter,
    _get_derivative_A,
    _tf2ss,
    FIRuncFilter,
    IIRuncFilter,
)
from PyDynamic.uncertainty.propagate_MonteCarlo import MC
from .conftest import random_array
from ..conftest import random_covariance_matrix


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
