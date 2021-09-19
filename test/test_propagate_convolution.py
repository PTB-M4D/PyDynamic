"""Test PyDynamic.uncertainty.propagate_convolve"""
from typing import List, Optional, Set

import numpy as np
import pytest
import scipy.ndimage as sn
from hypothesis import assume, given, strategies as hst

from PyDynamic.uncertainty.propagate_convolution import convolve_unc
from .conftest import random_covariance_matrix


def random_array(length):
    array = np.random.randn(length)
    return array


def valid_inputs(reduced_set: bool = False) -> List[List[np.ndarray]]:

    list_of_valid_inputs = []

    for n in [10, 15, 20]:
        x_signal = random_array(n)
        u_signal = random_covariance_matrix(n)

        if reduced_set:
            list_of_valid_inputs.append([x_signal, u_signal])
        else:
            list_of_valid_inputs.append([x_signal, None])
            list_of_valid_inputs.append([x_signal, np.diag(np.diag(u_signal))])
            list_of_valid_inputs.append([x_signal, u_signal])

    return list_of_valid_inputs


def valid_modes(restrict_kind_to: Optional[str] = None) -> Set[str]:
    scipy_modes = {"nearest", "reflect", "mirror"}
    numpy_modes = {"full", "valid", "same"}

    if restrict_kind_to == "scipy":
        return scipy_modes
    elif restrict_kind_to == "numpy":
        return numpy_modes
    else:
        return numpy_modes.union(scipy_modes)


@pytest.mark.parametrize("input_1", valid_inputs())
@pytest.mark.parametrize("input_2", valid_inputs())
@pytest.mark.parametrize("mode", valid_modes())
@pytest.mark.slow
def test_convolution(input_1, input_2, mode):
    # calculate the convolution of x1 and x2
    y, Uy = convolve_unc(*input_1, *input_2, mode)

    if mode in valid_modes("numpy"):
        y_ref = np.convolve(input_1[0], input_2[0], mode=mode)
    else:  # mode in valid_modes("scipy"):
        y_ref = sn.convolve1d(input_1[0], input_2[0], mode=mode)

    # compare results
    assert len(y) == len(Uy)
    assert len(y) == len(y_ref)
    assert np.allclose(y, y_ref)


def test_convolution_common_call():
    # check common execution of convolve_unc
    assert convolve_unc(
        *valid_inputs(reduced_set=True)[0], *valid_inputs(reduced_set=True)[0]
    )


@pytest.mark.parametrize("input_1", valid_inputs(reduced_set=True))
@pytest.mark.parametrize("input_2", valid_inputs(reduced_set=True))
@pytest.mark.parametrize("mode", valid_modes())
@pytest.mark.slow
def test_convolution_monte_carlo(input_1, input_2, mode):
    y, Uy = convolve_unc(*input_1, *input_2, mode)

    n_runs = 40000
    XX1 = np.random.multivariate_normal(*input_1, size=n_runs)
    XX2 = np.random.multivariate_normal(*input_2, size=n_runs)
    if mode in valid_modes("numpy"):
        convolve = np.convolve
    else:  # mode in valid_modes("scipy"):
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

    assert np.allclose(y, y_mc, rtol=1e-1, atol=1e-1)
    assert np.allclose(Uy, Uy_mc, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("input_1", valid_inputs(reduced_set=True))
@pytest.mark.parametrize("input_2", valid_inputs(reduced_set=True))
@given(hst.text())
def test_convolution_invalid_mode(input_1, input_2, mode):
    set_of_valid_modes = set(valid_modes())
    assume(mode not in set_of_valid_modes)

    with pytest.raises(ValueError):
        convolve_unc(*input_1, *input_2, mode)
