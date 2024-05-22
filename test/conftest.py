import os
from inspect import getmodule, stack
from math import ceil
from typing import Any, Callable, cast, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pytest
import scipy.stats as stats
from hypothesis import assume, HealthCheck, settings, strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy.linalg import LinAlgError
from scipy.signal import correlate

from PyDynamic import make_semiposdef
from PyDynamic.misc.tools import normalize_vector_or_matrix

custom_atol = 1e2 * np.finfo(np.float64).eps

def _check_for_ci_to_switch_off_performance_related_healthchecks():
    if _running_in_ci():
        _register_and_load_hypothesis_profile_for_slow_ci_machines()


def _running_in_ci():
    return os.getenv("CIRCLECI") == "true"


def _register_and_load_hypothesis_profile_for_slow_ci_machines():
    settings.register_profile(
        name="ci", suppress_health_check=(HealthCheck.too_slow,), deadline=None
    )
    settings.load_profile("ci")


_check_for_ci_to_switch_off_performance_related_healthchecks()


def _print_during_test_to_avoid_timeout(capsys):
    with capsys.disabled():
        frame = stack()[1]
        print(
            f"Run iteration of `{getmodule(frame[0]).__file__}::" f"{frame.function}`."
        )


def _is_np_array(array: Any) -> bool:
    return type(array).__module__ == np.__name__


def _is_np_array_with_len_greater_zero(array: Any) -> bool:
    return _is_np_array(array) and len(array) > 0


@composite
def FIRuncFilter_input(
    draw: Callable, exclude_corr_kind: bool = False
) -> Dict[str, Any]:
    min_accepted_by_scipy_linalg_companion = 2
    filter_length = draw(
        hst.integers(min_value=min_accepted_by_scipy_linalg_companion, max_value=100)
    )
    filter_theta = draw(
        hypothesis_float_vector(length=filter_length, min_value=1e-2, max_value=1e2)
    )
    filter_theta_covariance = draw(
        hst.one_of(
            hypothesis_covariance_matrix(
                number_of_rows=filter_length, max_value=10 * np.max(filter_theta)
            ),
            hst.just(None),
        )
    )

    signal_length = draw(hst.integers(min_value=200, max_value=1000))
    signal = draw(
        hypothesis_float_vector(length=signal_length, min_value=1e-2, max_value=1e3)
    )

    if exclude_corr_kind:
        kind = draw(hst.sampled_from(("float", "diag")))
    else:
        kind = draw(hst.sampled_from(("float", "diag", "corr")))
    if kind == "diag":
        signal_standard_deviation = draw(
            hypothesis_float_vector(
                length=signal_length, min_value=0, max_value=10 * np.max(signal)
            )
        )
    elif kind == "corr":
        random_data = draw(
            hypothesis_float_vector(
                length=signal_length // 2, min_value=1e-2, max_value=1e2
            )
        )
        acf = correlate(random_data, random_data, mode="full")

        scaled_acf = scale_matrix_or_vector_to_range(
            acf, range_min=1e-10, range_max=1e3
        )
        signal_standard_deviation = scaled_acf[len(scaled_acf) // 2 :]
    else:
        signal_standard_deviation = draw(
            hypothesis_not_negative_float(max_value=10 * np.max(signal))
        )

    lowpass_length = draw(
        hst.integers(min_value=min_accepted_by_scipy_linalg_companion, max_value=10)
    )
    lowpass = draw(
        hst.one_of(
            (
                hypothesis_float_vector(
                    length=lowpass_length, min_value=1e-2, max_value=1e2
                ),
                hst.just(None),
            )
        )
    )

    shift = draw(hypothesis_not_negative_float(max_value=1e2))

    return {
        "theta": filter_theta,
        "Utheta": filter_theta_covariance,
        "y": signal,
        "sigma_noise": signal_standard_deviation,
        "kind": kind,
        "blow": lowpass,
        "shift": shift,
    }


def check_no_nans_and_infs(*args: Tuple[np.ndarray]) -> bool:
    no_nans_and_infs = [np.all(np.isfinite(ndarray)) for ndarray in args]
    return np.all(no_nans_and_infs)


class VectorAndCompatibleMatrix(NamedTuple):
    vector: np.ndarray
    matrix: np.ndarray


@composite
def hypothesis_dimension(
    draw: Callable, min_value: Optional[int] = None, max_value: Optional[int] = None
) -> SearchStrategy:
    return (
        min_value
        if min_value is not None and min_value == max_value
        else draw(
            hst.one_of(
                hypothesis_even_dimension(min_value=min_value, max_value=max_value),
                hypothesis_odd_dimension(min_value=min_value, max_value=max_value),
            )
        )
    )


@composite
def hypothesis_even_dimension(
    draw: Callable, min_value: Optional[int] = None, max_value: Optional[int] = None
) -> SearchStrategy:
    minimum_dimension = min_value if min_value is not None else 2
    maximum_dimension = max_value if max_value is not None else 20
    even_dimension = (
        minimum_dimension
        if minimum_dimension is not None
        and minimum_dimension % 2 == 0
        and minimum_dimension == maximum_dimension
        else draw(
            hst.integers(
                min_value=ceil(minimum_dimension / 2), max_value=maximum_dimension // 2
            )
        )
        * 2
    )
    assert (
        minimum_dimension <= even_dimension <= maximum_dimension
        and even_dimension % 2 == 0
    )
    return even_dimension


@composite
def hypothesis_odd_dimension(
    draw: Callable, min_value: Optional[int] = None, max_value: Optional[int] = None
) -> SearchStrategy:
    minimum_dimension = min_value if min_value is not None else 1
    maximum_dimension = max_value if max_value is not None else 19
    odd_dimension = (
        minimum_dimension
        if minimum_dimension is not None
        and minimum_dimension % 2 == 1
        and minimum_dimension == maximum_dimension
        else draw(
            hypothesis_even_dimension(
                min_value=minimum_dimension - 1, max_value=maximum_dimension - 1
            )
        )
        + 1
    )
    assert (
        minimum_dimension <= odd_dimension <= maximum_dimension
        and odd_dimension % 2 == 1
    )
    return odd_dimension


@composite
def hypothesis_two_dimensional_array_shape(
    draw: Callable,
    ensure_even_second_dimension: bool = False,
    ensure_odd_second_dimension: bool = False,
):
    if ensure_even_second_dimension:
        second_dimension = draw(hypothesis_even_dimension())
    elif ensure_odd_second_dimension:
        second_dimension = draw(hypothesis_odd_dimension())
    else:
        second_dimension = draw(hypothesis_dimension())
    return draw(hypothesis_dimension()), second_dimension


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
        number_of_rows if number_of_rows is not None else draw(hypothesis_dimension())
    )
    return draw(
        hypothesis_float_square_matrix_strategy(
            number_of_rows=number_of_rows_and_columns
        )
    )


@composite
def hypothesis_nonzero_complex_vector(
    draw: DrawFn,
    length: Optional[int] = None,
    min_magnitude: float = 1e-4,
    max_magnitude: float = 1e4,
) -> SearchStrategy[np.ndarray]:
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
    return cast(SearchStrategy[np.ndarray], complex_vector)


@pytest.fixture
def random_complex_vector() -> Callable:
    def create_random_complex_vector(length: int) -> np.ndarray:
        return np.random.random(length) + 1j * np.random.random(length)

    return create_random_complex_vector


@composite
def hypothesis_float_vector(
    draw: DrawFn,
    length: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    exclude_min: bool = False,
    exclude_max: bool = False,
) -> SearchStrategy[np.ndarray]:
    number_of_elements = draw(hypothesis_dimension(min_value=length, max_value=length))
    return cast(
        SearchStrategy[np.ndarray],
        draw(
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
        ),
    )


def hypothesis_not_negative_float_strategy(
    max_value: Optional[float] = None,
    allow_nan: bool = False,
    allow_infinity: bool = False,
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
    exclude_min: bool = False,
    exclude_max: bool = False,
    allow_infinity: bool = False,
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
    draw: DrawFn,
    max_value: Optional[float] = None,
    allow_nan: bool = False,
    allow_infinity: bool = False,
) -> SearchStrategy[float]:
    return cast(
        SearchStrategy[float],
        draw(
            hypothesis_not_negative_float_strategy(
                max_value=max_value, allow_nan=allow_nan, allow_infinity=allow_infinity
            )
        ),
    )


@composite
def hypothesis_bounded_float(
    draw: DrawFn,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    exclude_min: bool = False,
    exclude_max: bool = False,
    allow_infinity: bool = False,
) -> SearchStrategy[float]:
    return cast(
        SearchStrategy[float],
        draw(
            hypothesis_bounded_float_strategy(
                min_value=min_value,
                max_value=max_value,
                exclude_min=exclude_min,
                exclude_max=exclude_max,
                allow_infinity=allow_infinity,
            )
        ),
    )


@composite
def hypothesis_covariance_matrix(
    draw: DrawFn,
    number_of_rows: Optional[int] = None,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> SearchStrategy[np.ndarray]:
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
    return cast(SearchStrategy[np.ndarray], scaled_cov)


@composite
def ensure_hypothesis_nonzero_diagonal(
    draw: DrawFn, square_matrix: np.ndarray
) -> SearchStrategy[np.ndarray]:
    return cast(
        SearchStrategy[np.ndarray],
        square_matrix
        + np.diag(
            draw(
                hypothesis_float_vector(
                    length=len(square_matrix), min_value=0, exclude_min=True
                )
            )
        ),
    )


def scale_matrix_or_vector_to_range(
    array: np.ndarray, range_min: float = 0.0, range_max: float = 1.0
) -> np.ndarray:
    return normalize_vector_or_matrix(array) * (range_max - range_min) + range_min


def scale_matrix_or_vector_to_convex_combination(array: np.ndarray) -> np.ndarray:
    return array / np.sum(array)


@composite
def hypothesis_covariance_matrix_with_zero_correlation(
    draw: DrawFn, number_of_rows: Optional[int] = None
) -> SearchStrategy[np.ndarray]:
    cov = np.diag(
        np.diag(draw(hypothesis_covariance_matrix(number_of_rows=number_of_rows)))
    )
    assume(np.all(np.linalg.eigvals(cov) >= 0))
    return cast(SearchStrategy[np.ndarray], cov)


@composite
def hypothesis_covariance_matrix_for_complex_vectors(
    draw: DrawFn,
    length: int,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> SearchStrategy[np.ndarray]:

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
    return cast(SearchStrategy[np.ndarray], uy_positive_semi_definite)


def random_covariance_matrix(length: int) -> np.ndarray:
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
def hypothesis_positive_powers_of_two(
    draw: DrawFn, min_k: int = 0, max_k: Optional[int] = None
) -> SearchStrategy[int]:
    k = draw(hst.integers(min_value=min_k, max_value=max_k))
    two_to_the_k = 2**k
    return cast(SearchStrategy[int], two_to_the_k)


@pytest.fixture
def corrmatrix() -> Callable:
    def _create_corrmatrix(
        rho: float, Nx: int, nu: float = 0.5, phi: float = 0.3
    ) -> np.ndarray:
        """Additional helper function to create a correlation matrix"""
        corrmat = np.zeros((Nx, Nx))
        if rho > 1:
            raise ValueError("Correlation scalar should be less than one.")

        for k in range(1, Nx):
            corrmat += np.diag(np.ones(Nx - k) * rho ** (phi * k**nu), k)
        corrmat += corrmat.T
        corrmat += np.eye(Nx)

        return corrmat

    return _create_corrmatrix
