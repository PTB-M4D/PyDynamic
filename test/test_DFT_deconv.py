"""Test PyDynamic.uncertainty.propagate_DFT.DFT_deconv"""
import sys

import numpy as np
import pytest
import scipy.stats as stats
from hypothesis import given, HealthCheck, settings
from numpy.testing import assert_allclose

from PyDynamic import DFT_deconv
from PyDynamic.misc.tools import progress_bar
from .conftest import random_covariance_matrix, two_to_the_k


@pytest.mark.slow
@given(two_to_the_k(min_k=4, max_k=6))
@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_dft_deconv(capsys, n):
    covariance_scale_minimizer = 1e-4
    h_complex = np.random.random(n) + 1j * np.random.random(n)
    h = np.r_[h_complex.real, h_complex.imag]
    y_complex = np.random.random(n) + 1j * np.random.random(n)
    y = np.r_[y_complex.real, y_complex.imag]
    uy = covariance_scale_minimizer * random_covariance_matrix(length=len(y))
    uh = covariance_scale_minimizer * random_covariance_matrix(length=len(h))
    x_deconv, u_deconv = DFT_deconv(H=h, Y=y, UH=uh, UY=uy)
    y_dist = stats.multivariate_normal(mean=y, cov=uy)
    h_dist = stats.multivariate_normal(mean=h, cov=uh)
    n = len(uh) - 2
    y_divided_by_h_mc = []
    n_monte_carlo_runs = 10000
    for i_monte_carlo_run in range(n_monte_carlo_runs):
        y_mc = y_dist.rvs()
        h_mc = h_dist.rvs()
        y_complex_mc = y_mc[: n // 2 + 1] + 1j * y_mc[n // 2 + 1 :]
        h_complex_mc = h_mc[: n // 2 + 1] + 1j * h_mc[n // 2 + 1 :]
        y_divided_by_h_mc.append(
            np.r_[
                np.real(y_complex_mc / h_complex_mc),
                np.imag(y_complex_mc / h_complex_mc),
            ]
        )
        with capsys.disabled():
            progress_bar(
                i_monte_carlo_run,
                n_monte_carlo_runs,
                prefix="Monte Carlo for test_dft_deconv() running:",
                fout=sys.stdout,
            )
    y_divided_by_h_mc_mean = np.mean(y_divided_by_h_mc, axis=0)
    y_divided_by_h_mc_cov = np.cov(y_divided_by_h_mc, rowvar=False)
    assert_allclose(x_deconv, y_divided_by_h_mc_mean, rtol=5e-3)
    assert_allclose(u_deconv, y_divided_by_h_mc_cov, rtol=35e1)
