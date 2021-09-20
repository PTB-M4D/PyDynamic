import os
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
import pytest
from hypothesis import assume, HealthCheck, settings, strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite, SearchStrategy

from PyDynamic import make_semiposdef

# This will check, if the testrun is executed in the ci environment and if so,
# disables the 'too_slow' health check. See
# https://hypothesis.readthedocs.io/en/latest/healthchecks.html#hypothesis.HealthCheck
# for some details.
settings.register_profile(
    name="ci", suppress_health_check=(HealthCheck.too_slow,), deadline=None
)
if "CIRCLECI" in os.environ:
    settings.load_profile("ci")


def check_no_nans_and_infs(*args: Tuple[np.ndarray]) -> bool:
    no_nans_and_infs = [np.all(np.isfinite(ndarray)) for ndarray in args]
    return np.all(no_nans_and_infs)


class VectorAndCompatibleMatrix(NamedTuple):
    vector: np.ndarray
    matrix: np.ndarray


@composite
def reasonable_random_dimension_strategy(draw: Callable):
    return draw(hst.integers(min_value=1, max_value=20))


def random_float_square_matrix_strategy(
    number_of_rows: int,
) -> SearchStrategy:
    return random_float_matrix_strategy(number_of_rows, number_of_rows)


def random_float_matrix_strategy(
    number_of_rows: int, number_of_cols: int
) -> SearchStrategy:
    return hnp.arrays(dtype=float, shape=(number_of_rows, number_of_cols))


@composite
def random_float_matrix(
    draw: Callable,
    number_of_rows: Optional[int] = None,
    number_of_cols: Optional[int] = None,
) -> np.ndarray:
    number_of_rows = (
        number_of_rows
        if number_of_rows is not None
        else draw(reasonable_random_dimension_strategy())
    )
    number_of_cols = (
        number_of_cols
        if number_of_cols is not None
        else draw(reasonable_random_dimension_strategy())
    )
    return draw(
        random_float_matrix_strategy(
            number_of_rows=number_of_rows, number_of_cols=number_of_cols
        )
    )


@composite
def random_float_square_matrix(
    draw: Callable, number_of_rows: Optional[int] = None
) -> np.ndarray:
    number_of_rows_and_columns = (
        number_of_rows
        if number_of_rows is not None
        else draw(hst.integers(min_value=1, max_value=20))
    )
    return draw(
        random_float_square_matrix_strategy(number_of_rows=number_of_rows_and_columns)
    )


@composite
def nonzero_complex_vector(draw: Callable, length: Optional[int] = None) -> np.ndarray:
    number_of_elements = (
        length if length is not None else draw(reasonable_random_dimension_strategy())
    )
    complex_vector = draw(
        hnp.arrays(
            dtype=complex,
            shape=number_of_elements,
            elements=hst.complex_numbers(
                min_magnitude=1e-4,
                max_magnitude=1e4,
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
) -> np.ndarray:
    number_of_elements = (
        length if length is not None else draw(reasonable_random_dimension_strategy())
    )
    return draw(
        hnp.arrays(
            dtype=float,
            elements=hst.floats(
                min_value=min_value,
                max_value=max_value,
                allow_infinity=False,
                allow_nan=False,
            ),
            shape=number_of_elements,
        )
    )


def random_not_negative_float_strategy() -> SearchStrategy:
    return hst.floats(min_value=0)


@composite
def random_not_negative_float(draw: Callable) -> float:
    return draw(random_not_negative_float_strategy)


@composite
def hypothesis_covariance_matrix(
    draw: Callable, number_of_rows: Optional[int] = None
) -> np.ndarray:
    number_of_rows_and_columns = (
        number_of_rows
        if number_of_rows is not None
        else draw(reasonable_random_dimension_strategy())
    )
    cov = np.cov(
        draw(
            hnp.arrays(
                dtype=float,
                elements=hst.floats(
                    min_value=0,
                    max_value=1,
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
    cov_after_discarding_smallest_singular_value = discard_smallest_singular_value(cov)
    assume(np.all(np.linalg.eigvals(cov_after_discarding_smallest_singular_value) >= 0))
    return cov_after_discarding_smallest_singular_value


@composite
def hypothesis_covariance_matrix_for_complex_vectors(
    draw: Callable,
    length: int,
) -> np.ndarray:

    uy_rr = draw(hypothesis_covariance_matrix(number_of_rows=length))
    uy_ii = draw(hypothesis_covariance_matrix(number_of_rows=length))
    uy_ri = draw(hypothesis_covariance_matrix(number_of_rows=length))
    uy = np.block([[uy_rr, uy_ri], [uy_ri.T, uy_ii]])
    assume(np.all(np.linalg.eigvals(uy) >= 0))
    return uy


def random_covariance_matrix(length: Optional[int]) -> np.ndarray:
    """Construct a valid (but random) covariance matrix with good condition number"""

    # because np.cov estimates the mean from data, the returned covariance matrix
    # has one eigenvalue close to numerical zero (rank n-1).
    # This leads to a singular matrix, which is badly suited to be used as valid
    # covariance matrix. To circumvent this:
    rng = np.random.default_rng()
    cov = np.cov(rng.random(size=(length + 1, length + 1)))
    cov_after_discarding_smallest_singular_value = discard_smallest_singular_value(cov)
    return cov_after_discarding_smallest_singular_value


def discard_smallest_singular_value(matrix: np.ndarray):
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
def two_to_the_k(
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
