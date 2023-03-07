from typing import Callable, Dict, List, Set, Tuple, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as hst
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import assume, given, HealthCheck, settings, Verbosity
from hypothesis.strategies import composite
from numpy.testing import assert_equal

from PyDynamic.misc.tools import (
    complex_2_real_imag,
    is_2d_matrix,
    is_2d_square_matrix,
    is_vector,
    normalize_vector_or_matrix,
    plot_vectors_and_covariances_comparison,
    real_imag_2_complex,
    separate_real_imag_of_mc_samples,
    separate_real_imag_of_vector,
    trimOrPad,
    trimOrPad_ND,
)
from .conftest import (
    hypothesis_covariance_matrix,
    hypothesis_dimension,
    hypothesis_even_dimension,
    hypothesis_float_vector,
    hypothesis_odd_dimension,
    hypothesis_two_dimensional_array_shape,
)


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


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hypothesis_dimension()))
def test_complex_2_real_imag_vector_len(array):
    assert_equal(len(complex_2_real_imag(array)), len(array) * 2)


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hypothesis_dimension()))
def test_complex_2_real_imag_vector_equality_real(array):
    array_real_imag = complex_2_real_imag(array)
    assert_equal(array_real_imag[: len(array_real_imag) // 2], np.real(array))


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hypothesis_dimension()))
def test_complex_2_real_imag_vector_equality_imag(array):
    array_real_imag = complex_2_real_imag(array)
    assert_equal(array_real_imag[len(array_real_imag) // 2 :], np.imag(array))


@given(
    hnp.arrays(
        dtype=hnp.scalar_dtypes(), shape=hypothesis_two_dimensional_array_shape()
    )
)
def test_complex_2_real_imag_array_len(array):
    array_real_imag = complex_2_real_imag(array)
    assert_equal(len(array_real_imag), len(array))
    assert_equal(array_real_imag.shape[1], array.shape[1] * 2)


@given(
    hnp.arrays(
        dtype=hnp.scalar_dtypes(), shape=hypothesis_two_dimensional_array_shape()
    )
)
def test_complex_2_real_imag_array_equality_real(array):
    array_real_imag = complex_2_real_imag(array)
    assert_equal(array_real_imag[:, : len(array_real_imag[0]) // 2], np.real(array))


@given(
    hnp.arrays(
        dtype=hnp.scalar_dtypes(), shape=hypothesis_two_dimensional_array_shape()
    )
)
def test_complex_2_real_imag_array_equality_imag(array):
    array_real_imag = complex_2_real_imag(array)
    assert_equal(array_real_imag[:, len(array_real_imag[0]) // 2 :], np.imag(array))


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hypothesis_odd_dimension()))
def test_separate_real_imag_of_vector_wrong_len(array):
    with pytest.raises(
        ValueError,
        match="separate_real_imag_of_vector: vector of real and imaginary parts is "
        "expected to contain exactly as many real as imaginary parts but is of "
        r"odd length=.*",
    ):
        separate_real_imag_of_vector(array)


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hypothesis_even_dimension()))
def test_separate_real_imag_of_vector_dimensions(vector):
    list_of_separated_real_imag = separate_real_imag_of_vector(vector)
    assert_equal(len(list_of_separated_real_imag), 2)
    assert_equal(
        _set_of_lens_of_list_entries(list_of_separated_real_imag),
        {len(vector) / 2},
    )


def _set_of_lens_of_list_entries(list_of_anything: List) -> Set[int]:
    return set(len(list_entry) for list_entry in list_of_anything)


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hypothesis_even_dimension()))
def test_separate_real_imag_of_vector_first_half(vector):
    assert_equal(separate_real_imag_of_vector(vector)[0], vector[: len(vector) // 2])


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hypothesis_even_dimension()))
def test_separate_real_imag_of_vector_second_half(vector):
    assert_equal(separate_real_imag_of_vector(vector)[1], vector[len(vector) // 2 :])


@given(
    hnp.arrays(
        dtype=hnp.scalar_dtypes(),
        shape=hypothesis_two_dimensional_array_shape(ensure_odd_second_dimension=True),
    )
)
def test_separate_real_imag_of_mc_samples_wrong_len(array):
    with pytest.raises(
        ValueError,
        match="separate_real_imag_of_mc_samples: vectors of real and imaginary "
        "parts are expected to contain exactly as many real as "
        r"imaginary parts but the first one is of odd length=.*",
    ):
        separate_real_imag_of_mc_samples(array)


@given(
    hnp.arrays(
        dtype=hnp.scalar_dtypes(),
        shape=hypothesis_two_dimensional_array_shape(ensure_even_second_dimension=True),
    )
)
def test_separate_real_imag_of_mc_samples_dimensions(array):
    list_of_separated_real_imag = separate_real_imag_of_mc_samples(array)
    assert_equal(len(list_of_separated_real_imag), 2)
    assert_equal(
        _set_of_shapes_of_ndarray_list(list_of_separated_real_imag),
        {(len(array), len(array[0]) / 2)},
    )


def _set_of_shapes_of_ndarray_list(
    list_of_anything: List[np.ndarray],
) -> Set[Tuple[int, int]]:
    return set(list_entry.shape for list_entry in list_of_anything)


@given(
    hnp.arrays(
        dtype=hnp.scalar_dtypes(),
        shape=hypothesis_two_dimensional_array_shape(ensure_even_second_dimension=True),
    )
)
def test_separate_real_imag_of_mc_samples_first_half(array):
    assert_equal(
        separate_real_imag_of_mc_samples(array)[0], array[:, : len(array[0]) // 2]
    )


@given(
    hnp.arrays(
        dtype=hnp.scalar_dtypes(),
        shape=hypothesis_two_dimensional_array_shape(ensure_even_second_dimension=True),
    )
)
def test_separate_real_imag_of_mc_samples_second_half(array):
    assert_equal(
        separate_real_imag_of_mc_samples(array)[1], array[:, len(array[0]) // 2 :]
    )


@given(
    hnp.arrays(
        dtype=hst.one_of(
            hnp.unsigned_integer_dtypes(), hnp.integer_dtypes(), hnp.floating_dtypes()
        ),
        shape=hypothesis_two_dimensional_array_shape(ensure_even_second_dimension=True),
    )
)
def test_real_imag_2_complex_array_shape(array):
    assert_equal(real_imag_2_complex(array).shape, (len(array), len(array[0]) // 2))


@given(
    hnp.arrays(
        dtype=hst.one_of(
            hnp.unsigned_integer_dtypes(), hnp.integer_dtypes(), hnp.floating_dtypes()
        ),
        shape=hypothesis_even_dimension(),
    )
)
def test_real_imag_2_complex_vector_len(array):
    assert_equal(len(real_imag_2_complex(array)), len(array) // 2)


@given(
    hnp.arrays(
        dtype=hst.one_of(
            hnp.unsigned_integer_dtypes(), hnp.integer_dtypes(), hnp.floating_dtypes()
        ),
        shape=hypothesis_two_dimensional_array_shape(ensure_even_second_dimension=True),
    )
)
def test_real_imag_2_complex_array_values(array):
    half_the_array_length = len(array[0]) // 2
    assert_equal(
        real_imag_2_complex(array),
        array[:, :half_the_array_length] + 1j * array[:, half_the_array_length:],
    )


@given(
    hnp.arrays(
        dtype=hst.one_of(
            hnp.unsigned_integer_dtypes(), hnp.integer_dtypes(), hnp.floating_dtypes()
        ),
        shape=hypothesis_even_dimension(),
    )
)
def test_real_imag_2_complex_vector_values(array):
    half_the_array_length = len(array) // 2
    assert_equal(
        real_imag_2_complex(array),
        array[:half_the_array_length] + 1j * array[half_the_array_length:],
    )


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hypothesis_odd_dimension()))
def test_real_imag_2_complex_vector_wrong_len(vector):
    with pytest.raises(
        ValueError,
        match="separate_real_imag_of_vector: vector of real and imaginary parts is "
        "expected to contain exactly as many real as imaginary parts but is of "
        r"odd length=.*",
    ):
        real_imag_2_complex(vector)


@given(
    hnp.arrays(
        dtype=hnp.scalar_dtypes(),
        shape=hypothesis_two_dimensional_array_shape(ensure_odd_second_dimension=True),
    )
)
def test_real_imag_2_complex_array_wrong_len(array):
    with pytest.raises(
        ValueError,
        match="separate_real_imag_of_mc_samples: vectors of real and imaginary "
        "parts are expected to contain exactly as many real as "
        r"imaginary parts but the first one is of odd length=.*",
    ):
        real_imag_2_complex(array)


def test_trimOrPad():
    N = 10
    a = np.arange(N)

    assert np.all(trimOrPad(a, 8) == a[:8])
    assert np.all(trimOrPad(a, 12) == np.r_[a, [0,0]])


def test_trimOrPad_ND():
    N = 10
    a = np.arange(N)

    # compare against old implementation
    assert np.all(trimOrPad(a, 8) == trimOrPad_ND(a, 8))
    assert np.all(trimOrPad(a, 12) == trimOrPad_ND(a, 12))

    # test multidim capabilities
    a = np.ones((3,4,5))
    b = trimOrPad_ND(a, length=(3,4,5))

    assert np.all(a == b)

    # test real/imag capabilities
    ri2c = real_imag_2_complex
    a = np.arange(N)
    tmp1 = ri2c(trimOrPad_ND(a, N+2, real_imag_type=True))
    tmp2 = trimOrPad_ND(ri2c(a), N+2, real_imag_type=False)

    assert np.all(tmp1 == tmp2)