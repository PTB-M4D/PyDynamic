"""Test PyDynamic.uncertainty.propagate_convolve"""
from typing import Callable, Optional, Set, Tuple

import numpy as np
import pytest
import scipy.ndimage as sn
from hypothesis import assume, given, settings, strategies as hst
from hypothesis.strategies import composite
from numpy.testing import assert_allclose
from PyDynamic.uncertainty.propagate_convolution import convolve_unc

from .conftest import (
    hypothesis_covariance_matrix,
    hypothesis_covariance_matrix_with_zero_correlation,
    hypothesis_float_vector,
    reasonable_random_dimension_strategy,
)


@composite
def x_and_Ux(
    draw: Callable, reduced_set: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    dim = reasonable_random_dimension_strategy(min_value=4, max_value=6)
    x = draw(hypothesis_float_vector(length=dim, min_value=-10, max_value=10))
    dim = len(x)
    if reduced_set:
        ux_strategies = hypothesis_covariance_matrix(number_of_rows=dim, max_value=1e-3)
    else:
        ux_strategies = hst.one_of(
            (
                hypothesis_covariance_matrix(number_of_rows=dim),
                hypothesis_covariance_matrix_with_zero_correlation(number_of_rows=dim),
                hst.just(None),
            )
        )
    ux = draw(ux_strategies)
    return x, ux


scipy_modes = ("nearest", "reflect", "mirror")
numpy_modes = ("full", "valid", "same")


@composite
def valid_modes(draw, restrict_kind_to: Optional[str] = None) -> Set[str]:
    if restrict_kind_to == "scipy":
        return draw(hst.sampled_from(scipy_modes))
    elif restrict_kind_to == "numpy":
        return draw(hst.sampled_from(numpy_modes))
    else:
        return draw(hst.sampled_from(numpy_modes + scipy_modes))


@given(x_and_Ux(), x_and_Ux(), valid_modes())
@pytest.mark.slow
def test_convolution(input_1, input_2, mode):
    # calculate the convolution of x1 and x2
    y, Uy = convolve_unc(*input_1, *input_2, mode)

    if mode in numpy_modes:
        y_ref = np.convolve(input_1[0], input_2[0], mode=mode)
    else:  # mode in valid_modes("scipy"):
        y_ref = sn.convolve1d(input_1[0], input_2[0], mode=mode)

    # compare results
    assert len(y) == len(Uy)
    assert len(y) == len(y_ref)
    assert_allclose(y + 1, y_ref + 1)


@given(x_and_Ux(reduced_set=True), x_and_Ux(reduced_set=True))
@pytest.mark.slow
def test_convolution_common_call(input_1, input_2):
    # check common execution of convolve_unc
    assert convolve_unc(*input_1, *input_2)


@given(x_and_Ux(reduced_set=True), x_and_Ux(reduced_set=True), valid_modes())
@pytest.mark.slow
@settings(deadline=None)
def test_convolution_monte_carlo(input_1, input_2, mode):
    y, Uy = convolve_unc(*input_1, *input_2, mode)

    n_runs = 40000
    XX1 = np.random.multivariate_normal(*input_1, size=n_runs)
    XX2 = np.random.multivariate_normal(*input_2, size=n_runs)
    if mode in numpy_modes:
        convolve = np.convolve
    else:  # mode in scipy_modes:
        convolve = sn.convolve1d
    mc_results = [convolve(x1, x2, mode=mode) for x1, x2 in zip(XX1, XX2)]
    y_mc = np.mean(mc_results, axis=0)
    Uy_mc = np.cov(mc_results, rowvar=False)

    # HACK: for visualization during debugging
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=4)
    # _min = min(Uy.min(), Uy_mc.min())
    # _max = max(Uy.max(), Uy_mc.max())
    # ax[0].plot(y, label="fir")
    # ax[0].plot(y_mc, label="mc")
    # ax[0].set_title("mode: {0}, x1: {1}, x2: {2}".format(mode, len(x1), len(x2)))
    # ax[0].legend()
    # ax[1].imshow(Uy, vmin=_min, vmax=_max)
    # ax[1].set_title("PyDynamic")
    # ax[2].imshow(Uy_mc, vmin=_min, vmax=_max)
    # ax[2].set_title("numpy MC")
    # img = ax[3].imshow(np.log(np.abs(Uy-Uy_mc)))
    # ax[3].set_title("log(abs(diff))")
    # fig.colorbar(img, ax=ax[3])
    # plt.show()
    # /HACK

    assert_allclose(y, y_mc, rtol=1e-1, atol=1e-1)
    assert_allclose(Uy, Uy_mc, rtol=1e-1, atol=1e-1)


@given(x_and_Ux(reduced_set=True), x_and_Ux(reduced_set=True), hst.text())
@pytest.mark.slow
def test_convolution_invalid_mode(input_1, input_2, mode):
    assume(mode not in numpy_modes and mode not in scipy_modes)
    with pytest.raises(ValueError):
        convolve_unc(*input_1, *input_2, mode)
