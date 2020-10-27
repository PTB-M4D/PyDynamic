"""Perform test for uncertainty.propagate_filter"""
import itertools
import copy

import numpy as np
import pytest

from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter


def random_array(length):
    array = np.random.randn(length)
    return array


def random_nonnegative_array(length):
    array = np.random.random(length)
    return array


def random_positive_semidefinite_matrix(length):
    matrix = np.random.random((length, length))
    matrix = make_semiposdef(matrix)
    return matrix


def valid_filters():
    N = np.random.randint(2, 100)  # N >= 2, see scipy.linalg.companion
    theta = random_array(N)

    valid_filters = [
        {"theta": theta, "Utheta": None},
        {"theta": theta, "Utheta": random_positive_semidefinite_matrix(N)},
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
    N = np.random.randint(2, 10)  # N >= 2, see scipy.linalg.companion
    blow = random_array(N)

    valid_lows = [
        {"blow": None},
        {"blow": blow},
    ]

    return valid_lows


@pytest.fixture
def equal_filters():
    equal_filters = valid_filters()

    equal_filters[1]["Utheta"] = 0.0 * equal_filters[1]["Utheta"]

    return equal_filters


@pytest.fixture
def equal_signals():
    equal_signals = valid_signals()

    # some shortcuts
    s = equal_signals[0]["sigma_noise"]
    N = equal_signals[0]["y"].size

    equal_signals[1]["sigma_noise"] = np.full(N, s)
    equal_signals[2]["sigma_noise"] = np.zeros(N)
    equal_signals[2]["sigma_noise"][0] = np.square(s)

    return equal_signals


@pytest.mark.parametrize("filters", valid_filters())
@pytest.mark.parametrize("signals", valid_signals())
@pytest.mark.parametrize("lowpasses", valid_lows())
def test_FIRuncFilter(filters, signals, lowpasses):
    # Check expected output for thinkable permutations of input parameters.
    y, Uy = FIRuncFilter(**filters, **signals, **lowpasses)
    assert len(y) == len(signals["y"])
    assert len(Uy) == len(signals["y"])


def test_FIRuncFilter_equality(equal_filters, equal_signals):
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


def test_IIRuncFilter():
    pass
