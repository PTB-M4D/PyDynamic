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


def random_semiposdef_matrix(length):
    matrix = np.random.random((length, length))
    matrix = make_semiposdef(matrix)
    return matrix


def valid_filters():
    N = np.random.randint(2, 100)  # N >= 2, see scipy.linalg.companion
    theta = random_array(N)

    valid_filters = [
        {"theta": theta, "Utheta": None},
        {"theta": theta, "Utheta": random_semiposdef_matrix(N)},
    ]

    return valid_filters


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
    N = np.random.randint(2, 100)
    theta = random_array(N)

    equal_filters = [
        {"theta": theta, "Utheta": None},
        {"theta": theta, "Utheta": np.zeros((N, N))},
    ]

    return equal_filters


@pytest.fixture
def equal_signals():
    # some shortcuts
    N = np.random.randint(100, 1000)
    signal = random_array(N)
    s = np.random.randn()

    equal_signals = [
        {"y": signal, "sigma_noise": s, "kind": "float"},
        {"y": signal, "sigma_noise": np.full(N, s), "kind": "diag"},
        {"y": signal, "sigma_noise": np.array([s**2] + [0]*(N-1)), "kind": "corr"},
    ]

    return equal_signals

@pytest.mark.parametrize("f", valid_filters())
@pytest.mark.parametrize("s", valid_signals())
@pytest.mark.parametrize("l", valid_lows())
def test_FIRuncFilter(f, s, l):

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
