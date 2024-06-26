"""Test PyDynamic.uncertainty.propagate_DFT.DFT_deconv"""
from typing import Callable, cast, Dict, Tuple

import numpy as np
import pytest
import scipy.stats as stats
from hypothesis import given, HealthCheck, settings
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy.testing import assert_allclose

from PyDynamic.uncertainty.propagate_DFT import DFT_deconv
from .conftest import (
    hypothesis_covariance_matrix_for_complex_vectors,
    hypothesis_float_vector,
    hypothesis_nonzero_complex_vector,
    hypothesis_positive_powers_of_two,
    custom_atol,
)


@composite
def deconvolution_input(
    draw: DrawFn, reveal_bug: bool = False
) -> SearchStrategy[Dict[str, np.ndarray]]:
    n = draw(hypothesis_positive_powers_of_two(min_k=2, max_k=4))
    if reveal_bug:
        y = np.r_[
            draw(hypothesis_float_vector(length=n, min_value=0.5, max_value=1.0)),
            np.zeros(n),
        ]
        uy = np.eye(N=n * 2)
        h = np.r_[
            draw(hypothesis_float_vector(length=n, min_value=0.5, max_value=1.0)),
            draw(hypothesis_float_vector(length=n, min_value=1000.0, max_value=1001.0)),
        ]
        uh = np.eye(N=n * 2)
    else:
        covariance_bounds = {"min_value": 1e-17, "max_value": 1e-11}
        vector_magnitude_bounds = {"min_magnitude": 1e-2, "max_magnitude": 1e2}
        y_complex = draw(
            hypothesis_nonzero_complex_vector(length=n, **vector_magnitude_bounds)
        )
        y = np.r_[y_complex.real, y_complex.imag]
        uy = draw(
            hypothesis_covariance_matrix_for_complex_vectors(
                length=n, **covariance_bounds
            )
        )
        h_complex = draw(
            hypothesis_nonzero_complex_vector(length=n, **vector_magnitude_bounds)
        )
        h = np.r_[h_complex.real, h_complex.imag]
        uh = draw(
            hypothesis_covariance_matrix_for_complex_vectors(
                length=n, **covariance_bounds
            ),
        )
    return cast(
        SearchStrategy[Dict[str, np.ndarray]], {"Y": y, "UY": uy, "H": h, "UH": uh}
    )


@given(deconvolution_input())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
    ],
    max_examples=100,
)
@pytest.mark.slow
def test_dft_deconv(
    multivariate_complex_monte_carlo, complex_deconvolution_on_sets, DFT_deconv_input
):
    x_deconv, u_deconv = DFT_deconv(**DFT_deconv_input)
    n_monte_carlo_runs = 40000
    monte_carlo_mean, monte_carlo_cov = multivariate_complex_monte_carlo(
        vectors=(DFT_deconv_input["Y"], DFT_deconv_input["H"]),
        covariance_matrices=(DFT_deconv_input["UY"], DFT_deconv_input["UH"]),
        n_monte_carlo_runs=n_monte_carlo_runs,
        operator=complex_deconvolution_on_sets,
    )
    x_deconv_shift_away_from_zero = 1 - np.min(x_deconv)
    u_deconv_shift_away_from_zero = 1 - np.min(u_deconv)
    assert_allclose(
        x_deconv + x_deconv_shift_away_from_zero,
        monte_carlo_mean + x_deconv_shift_away_from_zero,
        rtol=6.8e-2,
        atol=custom_atol,
    )
    assert_allclose(
        u_deconv + u_deconv_shift_away_from_zero,
        monte_carlo_cov + u_deconv_shift_away_from_zero,
        rtol=4.98e-1,
        atol=custom_atol,
    )


@pytest.fixture(scope="module")
def multivariate_complex_monte_carlo(
    _monte_carlo_samples_for_several_vectors,
    _mean_of_multivariate_monte_carlo_samples,
    _covariance_of_multivariate_monte_carlo_samples,
):
    def _perform_multivariate_complex_monte_carlo(
        vectors: Tuple[np.ndarray, np.ndarray],
        covariance_matrices: Tuple[np.ndarray, np.ndarray],
        n_monte_carlo_runs: int,
        operator: Callable,
    ):
        vectors_monte_carlo_samples = _monte_carlo_samples_for_several_vectors(
            vectors=vectors,
            covariance_matrices=covariance_matrices,
            n_monte_carlo_runs=n_monte_carlo_runs,
        )
        vectors_monte_carlo_samples_after_applying_operator = operator(
            *vectors_monte_carlo_samples
        )
        monte_carlo_samples_mean = _mean_of_multivariate_monte_carlo_samples(
            vectors_monte_carlo_samples_after_applying_operator
        )
        monte_carlo_samples_cov = _covariance_of_multivariate_monte_carlo_samples(
            vectors_monte_carlo_samples_after_applying_operator
        )
        return monte_carlo_samples_mean, monte_carlo_samples_cov

    return _perform_multivariate_complex_monte_carlo


@pytest.fixture(scope="module")
def _monte_carlo_samples_for_several_vectors():
    def _draw_monte_carlo_samples_for_several_vectors(
        vectors: Tuple[np.ndarray, ...],
        covariance_matrices: Tuple[np.ndarray, ...],
        n_monte_carlo_runs: int,
    ) -> Tuple[np.ndarray, ...]:
        tuple_of_monte_carlo_samples_for_several_vectors = tuple(
            stats.multivariate_normal.rvs(
                mean=vector, cov=covariance_matrix, size=n_monte_carlo_runs
            )
            for vector, covariance_matrix in zip(vectors, covariance_matrices)
        )
        return tuple_of_monte_carlo_samples_for_several_vectors

    return _draw_monte_carlo_samples_for_several_vectors


@pytest.fixture(scope="module")
def complex_deconvolution_on_sets():
    def _perform_complex_deconvolution_on_sets(
        y_mcs_complex: np.ndarray, h_mcs_complex: np.ndarray
    ):
        real_imag_divider = y_mcs_complex.shape[1] // 2
        y_mcs = (
            y_mcs_complex[..., :real_imag_divider]
            + 1j * y_mcs_complex[..., real_imag_divider:]
        )
        h_mcs = (
            h_mcs_complex[..., :real_imag_divider]
            + 1j * h_mcs_complex[..., real_imag_divider:]
        )
        y_mcs_divided_by_h_mcs_complex = y_mcs / h_mcs
        y_mcs_divided_by_h_mcs = np.concatenate(
            (
                np.real(y_mcs_divided_by_h_mcs_complex),
                np.imag(y_mcs_divided_by_h_mcs_complex),
            ),
            axis=1,
        )
        return y_mcs_divided_by_h_mcs

    return _perform_complex_deconvolution_on_sets


@pytest.fixture(scope="module")
def _mean_of_multivariate_monte_carlo_samples():
    def _compute_mean_of_multivariate_monte_carlo_samples(
        monte_carlo_samples: np.ndarray,
    ):
        y_divided_by_h_mc_mean = np.mean(monte_carlo_samples, axis=0)
        return y_divided_by_h_mc_mean

    return _compute_mean_of_multivariate_monte_carlo_samples


@pytest.fixture(scope="module")
def _covariance_of_multivariate_monte_carlo_samples():
    def _compute_covariance_of_multivariate_monte_carlo_samples(
        monte_carlo_samples: np.ndarray,
    ):
        y_divided_by_h_mc_cov = np.cov(monte_carlo_samples, rowvar=False)
        return y_divided_by_h_mc_cov

    return _compute_covariance_of_multivariate_monte_carlo_samples


@given(deconvolution_input(reveal_bug=True))
@settings(deadline=None)
@pytest.mark.slow
def test_reveal_bug_in_dft_deconv_up_to_1_9(
    multivariate_complex_monte_carlo, complex_deconvolution_on_sets, DFT_deconv_input
):
    # Fix a bug discovered by partners of Volker Wilkens. The bug and the process of
    # fixing it was processed and thus is documented in PR #220:
    # https://github.com/PTB-M4D/PyDynamic/pull/220
    u_deconv = DFT_deconv(**DFT_deconv_input)[1]
    n_monte_carlo_runs = 2000
    _, y_divided_by_h_mc_cov = multivariate_complex_monte_carlo(
        vectors=(DFT_deconv_input["Y"], DFT_deconv_input["H"]),
        covariance_matrices=(DFT_deconv_input["UY"], DFT_deconv_input["UH"]),
        n_monte_carlo_runs=n_monte_carlo_runs,
        operator=complex_deconvolution_on_sets,
    )
    assert_allclose(u_deconv + 1, y_divided_by_h_mc_cov + 1, atol=custom_atol)
