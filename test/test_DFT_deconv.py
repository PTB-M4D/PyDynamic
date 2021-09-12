"""Test PyDynamic.uncertainty.propagate_DFT.DFT_deconv"""
import numpy as np
import scipy.stats as stats
from hypothesis import given, settings
from hypothesis.strategies import data
from numpy.testing import assert_allclose

from PyDynamic.uncertainty.propagate_DFT import DFT_deconv
from .conftest import (
    hypothesis_covariance_matrix_for_complex_vectors,
    nonzero_complex_vector,
    two_to_the_k,
)


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
    real_complex_divider_index = 2 * n // 2
    y_mcs = (
        y_mc[..., :real_complex_divider_index]
        + 1j * y_mc[..., real_complex_divider_index:]
    )
    h_mcs = (
        h_mc[..., :real_complex_divider_index]
        + 1j * h_mc[..., real_complex_divider_index:]
    )
    y_divided_by_h_mc_complex = y_mcs / h_mcs
    y_divided_by_h_mc = np.concatenate(
        (np.real(y_divided_by_h_mc_complex), np.imag(y_divided_by_h_mc_complex)), axis=1
    )
    y_divided_by_h_mc_mean = np.mean(y_divided_by_h_mc, axis=0)
    y_divided_by_h_mc_cov = np.cov(y_divided_by_h_mc, rowvar=False)
    assert_allclose(x_deconv, y_divided_by_h_mc_mean, rtol=24e-3, atol=2e-2)
    assert_allclose(u_deconv, y_divided_by_h_mc_cov, rtol=1224, atol=2e-2)
