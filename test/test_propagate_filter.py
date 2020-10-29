"""Perform test for uncertainty.propagate_filter"""
import itertools

import numpy as np
import scipy
import pytest

from PyDynamic.misc.tools import make_semiposdef, trimOrPad
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter
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

    ## run FIR with MC and extract diagonal of returned covariance
    y_mc, uy_mc = MC(x, ux, b, a, Uab, blow=blow, runs=2000)
    uy_mc = np.sqrt(np.diag(uy_mc))

    
    # HACK: for visualization during debugging
    # import matplotlib.pyplot as plt
    # plt.plot(uy_fir, label="fir")
    # plt.plot(uy_mc, label="mc")
    # plt.legend()
    # plt.show()
    # /HACK

    # approximative comparison after swing-in of MC-result
    assert np.allclose(uy_fir[len(b) :], uy_mc[len(b) :], atol=1e-1, rtol=1e-1)


def test_IIRuncFilter():
    pass
