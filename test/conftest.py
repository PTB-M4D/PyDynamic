import os
from inspect import stack
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
import pytest
import scipy.stats as stats
from hypothesis import assume, HealthCheck, settings, strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite, SearchStrategy
from numpy.linalg import LinAlgError
from psutil import cpu_percent, virtual_memory

from PyDynamic import make_semiposdef
from PyDynamic.misc.tools import normalize_vector_or_matrix

# This will check, if the testrun is executed in the ci environment and if so,
# disables the 'too_slow' health check. See
# https://hypothesis.readthedocs.io/en/latest/healthchecks.html#hypothesis.HealthCheck
# for some details.

settings.register_profile(
    name="ci", suppress_health_check=(HealthCheck.too_slow,), deadline=None
)
if os.getenv("CIRCLECI") == "true":
    settings.load_profile("ci")


def _print_current_ram_usage(capsys):
    with capsys.disabled():
        print(
            f"Run iteration of `{stack()[1].function}()` with "
            f"{virtual_memory().percent}% of RAM used and "
            f"{cpu_percent()}% of CPU."
        )


def check_no_nans_and_infs(*args: Tuple[np.ndarray]) -> bool:
    no_nans_and_infs = [np.all(np.isfinite(ndarray)) for ndarray in args]
    return np.all(no_nans_and_infs)


class VectorAndCompatibleMatrix(NamedTuple):
    vector: np.ndarray
    matrix: np.ndarray


@composite
def hypothesis_reasonable_dimension_strategy(
    draw: Callable, min_value: Optional[int] = 1, max_value: Optional[int] = 20
):
    return draw(hst.integers(min_value=min_value, max_value=max_value))


@composite
def hypothesis_even_dimension_strategy(
    draw: Callable, min_value: Optional[int] = 2, max_value: Optional[int] = 20
):
    even_dimension = (
        draw(hst.integers(min_value=min_value // 2 + 1, max_value=max_value // 2 + 1))
        * 2
    )
    _ensure_odd_dimension_is_odd_and_in_bounds(max_value, min_value, even_dimension)
    return even_dimension


@composite
def hypothesis_odd_dimension_strategy(
    draw: Callable, min_value: Optional[int] = 1, max_value: Optional[int] = 21
):
    even_dimension = (
        draw(hst.integers(min_value=min_value // 2, max_value=max_value // 2)) * 2
    )
    odd_dimension = even_dimension + 1
    _ensure_odd_dimension_is_odd_and_in_bounds(max_value, min_value, odd_dimension)
    return odd_dimension


def hypothesis_float_square_matrix_strategy(
    number_of_rows: int,
) -> SearchStrategy:
    return hypothesis_float_matrix_strategy(number_of_rows, number_of_rows)


def hypothesis_float_matrix_strategy(
    number_of_rows: int, number_of_cols: int
) -> SearchStrategy:
    return hnp.arrays(dtype=float, shape=(number_of_rows, number_of_cols))


@composite
def hypothesis_float_matrix(
    draw: Callable,
    number_of_rows: Optional[int] = None,
    number_of_cols: Optional[int] = None,
) -> np.ndarray:
    number_of_rows = draw(
        hypothesis_dimension(min_value=number_of_rows, max_value=number_of_rows)
    )
    number_of_cols = draw(
        hypothesis_dimension(min_value=number_of_cols, max_value=number_of_cols)
    )
    return draw(
        hypothesis_float_matrix_strategy(
            number_of_rows=number_of_rows, number_of_cols=number_of_cols
        )
    )


@composite
def hypothesis_float_square_matrix(
    draw: Callable, number_of_rows: Optional[int] = None
) -> np.ndarray:
    number_of_rows_and_columns = (
        number_of_rows
        if number_of_rows is not None
        else draw(hst.integers(min_value=1, max_value=20))
    )
    return draw(
        hypothesis_float_square_matrix_strategy(
            number_of_rows=number_of_rows_and_columns
        )
    )


@composite
def hypothesis_nonzero_complex_vector(
    draw: Callable,
    length: Optional[int] = None,
    min_magnitude: Optional[float] = 1e-4,
    max_magnitude: Optional[float] = 1e4,
) -> np.ndarray:
    number_of_elements = draw(hypothesis_dimension(min_value=length, max_value=length))
    complex_vector = draw(
        hnp.arrays(
            dtype=complex,
            shape=number_of_elements,
            elements=hst.complex_numbers(
                min_magnitude=min_magnitude,
                max_magnitude=max_magnitude,
                allow_infinity=False,
                allow_nan=False,
            ),
        )
    )
    assume(np.all(np.real(complex_vector) != 0))
    return complex_vector


@pytest.fixture
def random_complex_vector() -> Callable:
    def create_random_complex_vector(length: int) -> np.ndarray:
        return np.random.random(length) + 1j * np.random.random(length)

    return create_random_complex_vector


@composite
def hypothesis_float_vector(
    draw: Callable,
    length: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    exclude_min: Optional[bool] = False,
    exclude_max: Optional[bool] = False,
) -> np.ndarray:
    number_of_elements = draw(hypothesis_dimension(min_value=length, max_value=length))
    return draw(
        hnp.arrays(
            dtype=float,
            elements=hst.floats(
                min_value=min_value,
                max_value=max_value,
                exclude_min=exclude_min,
                exclude_max=exclude_max,
                allow_infinity=False,
                allow_nan=False,
            ),
            shape=number_of_elements,
        )
    )


def hypothesis_not_negative_float_strategy(
    max_value: Optional[float] = None,
    allow_nan: Optional[bool] = False,
    allow_infinity: Optional[bool] = False,
) -> SearchStrategy:
    return hst.floats(
        min_value=0,
        max_value=max_value,
        allow_nan=allow_nan,
        allow_infinity=allow_infinity,
    )


def hypothesis_bounded_float_strategy(
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    exclude_min: Optional[bool] = False,
    exclude_max: Optional[bool] = False,
    allow_infinity: Optional[bool] = False,
) -> SearchStrategy:
    return hst.floats(
        min_value=min_value,
        max_value=max_value,
        exclude_min=exclude_min,
        exclude_max=exclude_max,
        allow_infinity=allow_infinity,
    )


@composite
def hypothesis_not_negative_float(
    draw: Callable,
    max_value: Optional[float] = None,
    allow_nan: Optional[bool] = False,
    allow_infinity: Optional[bool] = False,
) -> float:
    return draw(
        hypothesis_not_negative_float_strategy(
            max_value=max_value, allow_nan=allow_nan, allow_infinity=allow_infinity
        )
    )


@composite
def hypothesis_bounded_float(
    draw: Callable,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    exclude_min: Optional[bool] = False,
    exclude_max: Optional[bool] = False,
    allow_infinity: Optional[float] = False,
) -> float:
    return draw(
        hypothesis_bounded_float_strategy(
            min_value=min_value,
            max_value=max_value,
            exclude_min=exclude_min,
            exclude_max=exclude_max,
            allow_infinity=allow_infinity,
        )
    )


@composite
def hypothesis_covariance_matrix(
    draw: Callable,
    number_of_rows: Optional[int] = None,
    min_value: Optional[float] = 0,
    max_value: Optional[float] = 1,
) -> np.ndarray:
    number_of_rows_and_columns = draw(
        hypothesis_dimension(min_value=number_of_rows, max_value=number_of_rows)
    )
    cov_with_one_eigenvalue_close_to_zero = np.cov(
        draw(
            hnp.arrays(
                dtype=float,
                elements=hst.floats(
                    min_value=min_value,
                    max_value=max_value,
                    exclude_max=True,
                    allow_infinity=False,
                    allow_nan=False,
                ),
                shape=(
                    number_of_rows_and_columns + 1,
                    number_of_rows_and_columns + 1,
                ),
            )
        )
    )
    cov_after_discarding_smallest_singular_value = _discard_smallest_singular_value(
        cov_with_one_eigenvalue_close_to_zero
    )
    nonzero_diagonal_cov = draw(
        ensure_hypothesis_nonzero_diagonal(cov_after_discarding_smallest_singular_value)
    )
    scaled_cov = scale_matrix_or_vector_to_range(
        nonzero_diagonal_cov, range_min=min_value, range_max=max_value
    )
    assume(np.all(np.linalg.eigvals(scaled_cov) >= 0))
    return scaled_cov


@composite
def ensure_hypothesis_nonzero_diagonal(
    draw: Callable, square_matrix: np.ndarray
) -> np.ndarray:
    return square_matrix + np.diag(
        draw(
            hypothesis_float_vector(
                length=len(square_matrix), min_value=0, exclude_min=True
            )
        )
    )


def scale_matrix_or_vector_to_range(
    array: np.ndarray, range_min: Optional[float] = 0, range_max: Optional[float] = 1
) -> np.ndarray:
    return normalize_vector_or_matrix(array) * (range_max - range_min) + range_min


def scale_matrix_or_vector_to_convex_combination(array: np.ndarray) -> np.ndarray:
    return array / np.sum(array)


@composite
def hypothesis_covariance_matrix_with_zero_correlation(
    draw: Callable, number_of_rows: Optional[int] = None
) -> np.ndarray:
    cov = np.diag(np.diag(draw(hypothesis_covariance_matrix(number_of_rows))))
    assume(np.all(np.linalg.eigvals(cov) >= 0))
    return cov


@composite
def hypothesis_dimension(
    draw: Callable,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    minimum_dimension = min_value if min_value is not None else 1
    maximum_dimension = max_value if max_value is not None else 20
    dimension = (
        minimum_dimension
        if minimum_dimension is not None and minimum_dimension == maximum_dimension
        else draw(
            hypothesis_reasonable_dimension_strategy(
                min_value=minimum_dimension, max_value=maximum_dimension
            )
        )
    )
    assert minimum_dimension <= dimension <= maximum_dimension
    return dimension


@composite
def hypothesis_even_dimension(
    draw: Callable,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    minimum_dimension = min_value if min_value is not None else 2
    maximum_dimension = max_value if max_value is not None else 20
    even_dimension = (
        minimum_dimension
        if minimum_dimension is not None
        and minimum_dimension % 2 == 0
        and minimum_dimension == maximum_dimension
        else draw(
            hypothesis_even_dimension_strategy(
                min_value=minimum_dimension, max_value=maximum_dimension
            )
        )
    )
    _ensure_odd_dimension_is_odd_and_in_bounds(
        maximum_dimension, minimum_dimension, even_dimension
    )
    return even_dimension


def _ensure_even_dimension_is_even_and_in_bounds(
    max_val: int, min_val: int, odd_dimension: int
):
    assert min_val <= odd_dimension <= max_val
    assert odd_dimension % 2 == 0


@composite
def hypothesis_odd_dimension(
    draw: Callable,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    minimum_dimension = min_value if min_value is not None else 1
    maximum_dimension = max_value if max_value is not None else 21
    odd_dimension = (
        minimum_dimension
        if minimum_dimension is not None
        and minimum_dimension % 2 == 1
        and minimum_dimension == maximum_dimension
        else draw(
            hypothesis_odd_dimension_strategy(
                min_value=minimum_dimension, max_value=maximum_dimension
            )
        )
    )
    _ensure_odd_dimension_is_odd_and_in_bounds(
        maximum_dimension, minimum_dimension, odd_dimension
    )
    return odd_dimension


def _ensure_odd_dimension_is_odd_and_in_bounds(
    max_val: int, min_val: int, odd_dimension: int
):
    assert min_val <= odd_dimension <= max_val
    assert odd_dimension % 2 == 1


@composite
def hypothesis_covariance_matrix_for_complex_vectors(
    draw: Callable,
    length: int,
    min_value: Optional[float] = 0.0,
    max_value: Optional[float] = 1.0,
) -> np.ndarray:

    uy_rr = draw(
        hypothesis_covariance_matrix(
            number_of_rows=length, min_value=min_value, max_value=max_value
        )
    )
    uy_ii = draw(
        hypothesis_covariance_matrix(
            number_of_rows=length, min_value=min_value, max_value=max_value
        )
    )
    uy_ri = draw(
        hypothesis_covariance_matrix(
            number_of_rows=length, min_value=min_value, max_value=max_value
        )
    )
    uy = np.block([[uy_rr, uy_ri], [uy_ri.T, uy_ii]])
    uy_positive_semi_definite = make_semiposdef(uy)
    assume(np.all(np.linalg.eigvals(uy_positive_semi_definite) >= 0))
    return uy_positive_semi_definite


def random_covariance_matrix(length: Optional[int]) -> np.ndarray:
    """Construct a valid (but random) covariance matrix with good condition number"""

    # because np.cov estimates the mean from data, the returned covariance matrix
    # has one eigenvalue close to numerical zero (rank n-1).
    # This leads to a singular matrix, which is badly suited to be used as valid
    # covariance matrix. To circumvent this:
    rng = np.random.default_rng()
    cov_with_one_eigenvalue_close_to_zero = np.cov(
        rng.random(size=(length + 1, length + 1))
    )
    cov_after_discarding_smallest_singular_value = _discard_smallest_singular_value(
        cov_with_one_eigenvalue_close_to_zero
    )
    cov_positive_semi_definite = cov_after_discarding_smallest_singular_value
    while True:
        try:
            stats.multivariate_normal(cov=cov_positive_semi_definite)
            break
        except LinAlgError:
            cov_positive_semi_definite = make_semiposdef(cov_positive_semi_definite)
    return cov_positive_semi_definite


def _discard_smallest_singular_value(matrix: np.ndarray) -> np.ndarray:
    u, s, vh = np.linalg.svd(matrix, full_matrices=False, hermitian=True)
    cov_after_discarding_smallest_singular_value = (u[:-1, :-1] * s[:-1]) @ vh[:-1, :-1]
    return cov_after_discarding_smallest_singular_value


@pytest.fixture
def random_covariance_matrix_for_complex_vectors() -> Callable:
    def _create_random_covariance_matrix_for_complex_vectors(length: int) -> np.ndarray:
        uy_rr = make_semiposdef(random_covariance_matrix(length=length), maxiter=100)
        uy_ii = make_semiposdef(random_covariance_matrix(length=length), maxiter=100)
        uy_ri = make_semiposdef(random_covariance_matrix(length=length), maxiter=100)
        return np.block([[uy_rr, uy_ri], [uy_ri.T, uy_ii]])

    return _create_random_covariance_matrix_for_complex_vectors


@composite
def hypothesis_two_to_the_k(
    draw: Callable, min_k: Optional[int] = None, max_k: Optional[int] = None
) -> int:
    k = draw(hst.integers(min_value=min_k, max_value=max_k))
    return 2 ** k


@pytest.fixture
def corrmatrix() -> Callable:
    def _create_corrmatrix(
        rho: float, Nx: int, nu: Optional[float] = 0.5, phi: Optional[float] = 0.3
    ) -> np.ndarray:
        """Additional helper function to create a correlation matrix"""
        corrmat = np.zeros((Nx, Nx))
        if rho > 1:
            raise ValueError("Correlation scalar should be less than one.")

        for k in range(1, Nx):
            corrmat += np.diag(np.ones(Nx - k) * rho ** (phi * k ** nu), k)
        corrmat += corrmat.T
        corrmat += np.eye(Nx)

        return corrmat

    return _create_corrmatrix
