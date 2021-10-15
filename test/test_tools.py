from typing import Callable, Dict, Union

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import assume, given, HealthCheck, settings, Verbosity
from hypothesis.strategies import composite
from matplotlib import pyplot as plt
from numpy.testing import assert_equal

from PyDynamic.misc.tools import (
    is_2d_matrix,
    is_2d_square_matrix,
    is_vector,
    normalize_vector_or_matrix,
    plot_vectors_and_covariances_comparison,
)
from .conftest import hypothesis_covariance_matrix, hypothesis_float_vector


@composite
def similar_vectors_and_matching_covariances(
    draw: Callable,
) -> Dict[str, Union[np.ndarray, str]]:
    vector_1 = draw(hypothesis_float_vector())
    dimension = len(vector_1)
    vector_2 = draw(hypothesis_float_vector(length=dimension))
    covariance_1 = draw(hypothesis_covariance_matrix(number_of_rows=dimension))
    covariance_2 = draw(hypothesis_covariance_matrix(number_of_rows=dimension))
    return {
        "vector_1": vector_1,
        "vector_2": vector_2,
        "covariance_1": covariance_1,
        "covariance_2": covariance_2,
    }


@given(similar_vectors_and_matching_covariances())
@settings(
    deadline=None,
    verbosity=Verbosity.verbose,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
    ],
)
@pytest.mark.slow
def test_display_vectors_and_covariances_comparison(monkeypatch, valid_inputs):
    monkeypatch.setattr(plt, "show", lambda: None, raising=True)
    plot_vectors_and_covariances_comparison(**valid_inputs)


@given(hnp.arrays(dtype=hnp.array_dtypes(), shape=hnp.array_shapes()))
def test_is_2d_square_matrix(array):
    assert is_2d_square_matrix(array) == (
        len(array.shape) == 2 and array.shape[0] == array.shape[1]
    )


@given(hnp.arrays(dtype=hnp.array_dtypes(), shape=hnp.array_shapes()))
def test_is_2d_matrix(array):
    assert is_2d_matrix(array) == (len(array.shape) == 2)


@given(hnp.arrays(dtype=hnp.array_dtypes(), shape=hnp.array_shapes()))
def test_is_vector(array):
    assert is_vector(array) == (len(array.shape) == 1)


@given(hnp.arrays(dtype=hnp.floating_dtypes(), shape=hnp.array_shapes()))
def test_normalize_vector_or_matrix_for_correct_shape(array):
    normalized = normalize_vector_or_matrix(array)
    assert_equal(normalized.shape, array.shape)


@given(hnp.arrays(dtype=hnp.floating_dtypes(), shape=hnp.array_shapes()))
def test_normalize_vector_or_matrix_for_correct_range(array):
    normalized = normalize_vector_or_matrix(array)
    assume(_all_are_not_nan(normalized))
    assert _minimum_is_zero(normalized)
    if _not_all_entries_the_same(array):
        assert _maximum_is_one(normalized)


def _not_all_entries_the_same(array: np.ndarray) -> bool:
    return np.min(array) != np.max(array)


def _all_are_not_nan(array: np.ndarray) -> bool:
    return not np.any(np.isnan(array))


def _minimum_is_zero(array: np.ndarray) -> bool:
    return np.min(array) == 0


def _maximum_is_one(array: np.ndarray) -> bool:
    return np.max(array) == 1
