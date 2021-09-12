"""Test PyDynamic.uncertainty.propagate_DFT.DFT_deconv"""

import numpy as np
import scipy.stats as stats
from hypothesis import given, HealthCheck, settings
from numpy.testing import assert_allclose

from PyDynamic.uncertainty.propagate_DFT import DFT_deconv
from .conftest import two_to_the_k


@given(two_to_the_k(min_k=2, max_k=4))
@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_dft_deconv(
    capsys, random_complex_vector, random_covariance_matrix_for_complex_vectors, n
):
    covariance_scale_minimizer = 1e-3
    y_complex = random_complex_vector(n)
    y = np.r_[y_complex.real, y_complex.imag]
    uy = covariance_scale_minimizer * random_covariance_matrix_for_complex_vectors(n)
    h_complex = random_complex_vector(n)
    h = np.r_[h_complex.real, h_complex.imag]
    uh = covariance_scale_minimizer * random_covariance_matrix_for_complex_vectors(n)
    x_deconv, u_deconv = DFT_deconv(H=h, Y=y, UH=uh, UY=uy)
    n_monte_carlo_runs = 40000
    y_mc = stats.multivariate_normal.rvs(mean=y, cov=uy, size=n_monte_carlo_runs)
    h_mc = stats.multivariate_normal.rvs(mean=h, cov=uh, size=n_monte_carlo_runs)
    mean_tolerance = 1e-2
    cov_tolerance = 2561
    assert_allclose(actual=np.mean(y_mc, axis=0), desired=y, rtol=mean_tolerance)
    assert_allclose(actual=np.cov(y_mc, rowvar=False), desired=uy, rtol=cov_tolerance)
    assert_allclose(actual=np.mean(h_mc, axis=0), desired=h, rtol=mean_tolerance)
    assert_allclose(actual=np.cov(h_mc, rowvar=False), desired=uh, rtol=cov_tolerance)
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
    assert_allclose(x_deconv, y_divided_by_h_mc_mean, rtol=24e-3)
    assert_allclose(u_deconv, y_divided_by_h_mc_cov, rtol=640)
