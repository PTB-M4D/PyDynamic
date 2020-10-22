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


def random_positive_definite_matrix(length):
    matrix = np.random.random((length, length))
    matrix = make_semiposdef(matrix)
    return matrix


@pytest.fixture
def valid_filters():
    N = np.random.randint(1, 100)
    theta = random_array(N)

    valid_filters = [
        {"theta": theta, "Utheta": None},
        {"theta": theta, "Utheta": random_positive_definite_matrix(N)},
    ]

    return valid_filters


@pytest.fixture
def valid_signals():
    N = np.random.randint(100, 1000)
    signal = random_array(N)

    valid_signals = [
        # {"y": signal, "sigma_noise": None, "kind": "float"},
        {"y": signal, "sigma_noise": np.random.randn(), "kind": "float"},
        {"y": signal, "sigma_noise": random_nonnegative_array(N), "kind": "diag"},
        {"y": signal, "sigma_noise": random_nonnegative_array(N // 2), "kind": "corr"},
    ]

    return valid_signals


@pytest.fixture
def valid_lows():
    N = np.random.randint(1, 10)
    blow = random_array(N)

    valid_lows = [
        {"blow": None},
        {"blow": blow},
    ]

    return valid_lows


@pytest.fixture
def equal_filters(valid_filters):

    equal_filters = copy.copy(valid_filters)
    equal_filters[1]["Utheta"] = 0.0 * equal_filters[1]["Utheta"]

    return equal_filters


@pytest.fixture
def equal_signals(valid_signals):
    equal_signals = copy.copy(valid_signals)

    # some shortcuts
    s = valid_signals[0]["sigma_noise"]
    N = valid_signals[0]["y"].size

    equal_signals[1]["sigma_noise"] = np.full(N, s)
    equal_signals[2]["sigma_noise"] = np.zeros(N)
    equal_signals[2]["sigma_noise"][0] = np.square(s)

    return equal_signals


def test_FIRuncFilter(valid_filters, valid_signals, valid_lows):

    # run all combinations of filter and signals
    for (f, s, l) in itertools.product(valid_filters, valid_signals, valid_lows):
        y, Uy = FIRuncFilter(**f, **s, **l)
        assert len(y) == len(s["y"])
        assert len(Uy) == len(s["y"])


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
