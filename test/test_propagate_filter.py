"""Perform test for uncertainty.propagate_filter"""
import itertools

import numpy as np
import scipy
import pytest

from PyDynamic.misc.tools import make_semiposdef, trimOrPad
from PyDynamic.uncertainty.propagate_filter import (
    FIRuncFilter,
    FIRuncFilter_2,
    _fir_filter,
)
from PyDynamic.uncertainty.propagate_MonteCarlo import MC


def random_array(length):
    array = np.random.randn(length)
    return array


def random_nonnegative_array(length):
    array = np.random.random(length)
    return array


def random_covariance_matrix(length):
    # construct a valid (but random) covariance matrix
    cov = np.cov(np.random.random((length, length)))
    return cov


def valid_filters():
    N = np.random.randint(2, 100)  # scipy.linalg.companion requires N >= 2
    theta = random_array(N)

    valid_filters = [
        {"theta": theta, "Utheta": None},
        {"theta": theta, "Utheta": np.zeros((N, N))},
        {"theta": theta, "Utheta": random_covariance_matrix(N)},
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


@pytest.mark.parametrize("filters", valid_filters())
@pytest.mark.parametrize("signals", valid_signals())
@pytest.mark.parametrize("lowpasses", valid_lows())
def test_FIRuncFilter_2(filters, signals, lowpasses):
    # Compare output of both functions for thinkable permutations of input parameters.
    y, Uy = FIRuncFilter(**filters, **signals, **lowpasses)
    y2, Uy2 = FIRuncFilter_2(**filters, **signals, **lowpasses)

    # check output dimensions
    assert len(y2) == len(signals["y"])
    assert Uy2.shape == (len(signals["y"]), len(signals["y"]))

    # check value identity
    assert np.allclose(y, y2)
    assert np.allclose(Uy, np.sqrt(np.diag(Uy2)))


def test_fir_filter_MC_comparison():
    N_signal = np.random.randint(20, 25)
    x = random_array(N_signal)
    Ux = random_covariance_matrix(N_signal)

    N_theta = np.random.randint(2, 5)  # scipy.linalg.companion requires N >= 2
    theta = random_array(N_theta)  # scipy.signal.firwin(N_theta, 0.1)
    Utheta = random_covariance_matrix(N_theta)

    # run method
    y_fir, Uy_fir = _fir_filter(x, theta, Ux, Utheta, initial_conditions="zero")

    ## run FIR with MC and extract diagonal of returned covariance
    y_mc, Uy_mc = MC(x, Ux, theta, [1.0], Utheta, blow=None, runs=10000)

    # HACK: for visualization during debugging
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # ax[0].plot(y_fir, label="fir")
    # ax[0].plot(y_mc, label="mc")
    # ax[0].set_title("filter: {0}, signal: {1}".format(len(theta), len(x)))
    # ax[0].legend()
    # ax[1].imshow(Uy_fir)
    # ax[2].imshow(Uy_mc)
    # plt.show()
    # /HACK

    # approximate comparison
    assert Uy_fir.shape == Uy_mc.shape
    assert np.allclose(Uy_fir, Uy_mc, atol=1e-1, rtol=1e-1)


def test_IIRuncFilter():
    pass
