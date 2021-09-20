"""Test PyDynamic.uncertainty.propagate_DFT.DFT_deconv"""
import numpy as np
import pytest
import scipy.stats as stats
from hypothesis import given, settings
from hypothesis.strategies import data
from numpy.testing import assert_allclose

from PyDynamic.uncertainty.propagate_DFT import DFT_deconv
from .conftest import (
    hypothesis_covariance_matrix_for_complex_vectors,
    hypothesis_float_vector,
    nonzero_complex_vector,
    two_to_the_k,
)


@pytest.mark.slow
@given(data(), two_to_the_k(min_k=2, max_k=4))
@settings(deadline=None)
def test_dft_deconv(
    hypothesis,
    n,
):
    covariance_scale_minimizer = 1e-3
    y_complex = hypothesis.draw(nonzero_complex_vector(length=n))
    y = np.r_[y_complex.real, y_complex.imag]
    uy = covariance_scale_minimizer * hypothesis.draw(
        hypothesis_covariance_matrix_for_complex_vectors(n)
    )
    h_complex = hypothesis.draw(nonzero_complex_vector(length=n))
    h = np.r_[h_complex.real, h_complex.imag]
    uh = covariance_scale_minimizer * hypothesis.draw(
        hypothesis_covariance_matrix_for_complex_vectors(n)
    )
    x_deconv, u_deconv = DFT_deconv(H=h, Y=y, UH=uh, UY=uy)
    n_monte_carlo_runs = 40000
    y_mc = stats.multivariate_normal.rvs(mean=y, cov=uy, size=n_monte_carlo_runs)
    h_mc = stats.multivariate_normal.rvs(mean=h, cov=uh, size=n_monte_carlo_runs)
    y_mcs = y_mc[..., :n] + 1j * y_mc[..., n:]
    h_mcs = h_mc[..., :n] + 1j * h_mc[..., n:]
    y_mcs_divided_by_h_mcs_complex = y_mcs / h_mcs
    y_mcs_divided_by_h_mcs = np.concatenate(
        (
            np.real(y_mcs_divided_by_h_mcs_complex),
            np.imag(y_mcs_divided_by_h_mcs_complex),
        ),
        axis=1,
    )
    y_divided_by_h_mc_mean = np.mean(y_mcs_divided_by_h_mcs, axis=0)
    y_divided_by_h_mc_cov = np.cov(y_mcs_divided_by_h_mcs, rowvar=False)
    assert_allclose(x_deconv + 1, y_divided_by_h_mc_mean + 1, rtol=3e-4, atol=4e-7)
    assert_allclose(u_deconv + 1, y_divided_by_h_mc_cov + 1, atol=4e-7)


@given(data(), two_to_the_k(min_k=2, max_k=4))
@settings(deadline=None)
def test_reveal_bug_in_dft_deconv_up_to_1_9(
    hypothesis,
    n,
):
    # Fix a bug discovered by partners of Volker Wilkens. The bug and the process of
    # fixing it was processed and thus is documented in PR #220:
    # https://github.com/PTB-M4D/PyDynamic/pull/220
    y = np.r_[
        hypothesis.draw(
            hypothesis_float_vector(length=n, min_value=0.5, max_value=1.0)
        ),
        np.zeros(n),
    ]
    uy = np.eye(N=n * 2)
    h = np.r_[
        hypothesis.draw(
            hypothesis_float_vector(length=n, min_value=0.5, max_value=1.0)
        ),
        hypothesis.draw(
            hypothesis_float_vector(length=n, min_value=1000.0, max_value=1001.0)
        ),
    ]
    uh = np.eye(N=n * 2)
    u_deconv = DFT_deconv(H=h, Y=y, UH=uh, UY=uy)[1]
    n_monte_carlo_runs = 2000
    y_mc = stats.multivariate_normal.rvs(mean=y, cov=uy, size=n_monte_carlo_runs)
    h_mc = stats.multivariate_normal.rvs(mean=h, cov=uh, size=n_monte_carlo_runs)
    y_mcs = y_mc[..., :n] + 1j * y_mc[..., n:]
    h_mcs = h_mc[..., :n] + 1j * h_mc[..., n:]
    y_mcs_divided_by_h_mcs_complex = y_mcs / h_mcs
    y_mcs_divided_by_h_mcs = np.concatenate(
        (
            np.real(y_mcs_divided_by_h_mcs_complex),
            np.imag(y_mcs_divided_by_h_mcs_complex),
        ),
        axis=1,
    )
    y_divided_by_h_mc_cov = np.cov(y_mcs_divided_by_h_mcs, rowvar=False)
    assert_allclose(u_deconv + 1, y_divided_by_h_mc_cov + 1)
