"""Test PyDynamic.uncertainty.propagate_convolve"""

import numpy as np
import pytest
import scipy.ndimage as sn
from hypothesis import assume, given, strategies as hst

from PyDynamic.uncertainty.propagate_convolution import convolve_unc


def random_array(length):
    array = np.random.randn(length)
    return array


def random_covariance_matrix(length):
    """construct a valid (but random) covariance matrix with good condition number"""

    # because np.cov estimates the mean from data, the returned covariance matrix
    # has one eigenvalue close to numerical zero (rank n-1).
    # This leads to a singular matrix, which is badly suited to be used as valid
    # covariance matrix. To circumvent this:

    ## draw random (n+1, n+1) matrix
    cov = np.cov(np.random.random((length + 1, length + 1)))

    ## calculate SVD
    u, s, vh = np.linalg.svd(cov, full_matrices=False, hermitian=True)

    ## reassemble a covariance of size (n, n) by discarding the smallest singular value
    cov_adjusted = (u[:-1, :-1] * s[:-1]) @ vh[:-1, :-1]

    return cov_adjusted


def valid_inputs(reduced_set=False):

    valid_inputs = []

    for n in [10, 15, 20]:
        x_signal = random_array(n)
        u_signal = random_covariance_matrix(n)

        if reduced_set:
            valid_inputs.append([x_signal, u_signal])
        else:
            valid_inputs.append([x_signal, None])
            valid_inputs.append([x_signal, np.diag(np.diag(u_signal))])
            valid_inputs.append([x_signal, u_signal])

    return valid_inputs


def valid_modes(kind="all"):
    scipy_modes = {"nearest", "reflect", "mirror"}
    numpy_modes = {"full", "valid", "same"}

    if kind == "all":
        return numpy_modes.union(scipy_modes)
    elif kind == "scipy":
        return scipy_modes
    elif kind == "numpy":
        return numpy_modes
    else:
        return set()


@pytest.mark.parametrize("input_1", valid_inputs())
@pytest.mark.parametrize("input_2", valid_inputs())
@pytest.mark.parametrize("mode", valid_modes())
def test_convolution(input_1, input_2, mode):

    scipy_modes = valid_modes("scipy")
    numpy_modes = valid_modes("numpy")

    # calculate the convolution of x1 and x2
    y, Uy = convolve_unc(*input_1, *input_2, mode)

    if mode in numpy_modes:
        y_ref = np.convolve(input_1[0], input_2[0], mode=mode)
    elif mode in scipy_modes:
        y_ref = sn.convolve1d(input_1[0], input_2[0], mode=mode)

    # compare results
    assert len(y) == len(Uy)
    assert len(y) == len(y_ref)
    assert np.allclose(y, y_ref)


@pytest.mark.parametrize("input_1", valid_inputs(reduced_set=True))
@pytest.mark.parametrize("input_2", valid_inputs(reduced_set=True))
@pytest.mark.parametrize("mode", valid_modes())
def test_convolution_monte_carlo(input_1, input_2, mode):

    scipy_modes = valid_modes("scipy")
    numpy_modes = valid_modes("numpy")

    # pydynamic calculation
    y, Uy = convolve_unc(*input_1, *input_2, mode)

    # Monte Carlo simulation
    mc_results = []
    n_runs = 20000
    XX1 = np.random.multivariate_normal(*input_1, size=n_runs)
    XX2 = np.random.multivariate_normal(*input_2, size=n_runs)
    for x1, x2 in zip(XX1, XX2):
        if mode in numpy_modes:
            conv = np.convolve(x1, x2, mode=mode)
        elif mode in scipy_modes:
            conv = sn.convolve1d(x1, x2, mode=mode)
        mc_results.append(conv)

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
