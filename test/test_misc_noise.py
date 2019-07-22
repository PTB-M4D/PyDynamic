# -*- coding: utf-8 -*-
"""
Perform tests on identification sub-packages.

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pytest

# import PyDynamic from local code, not from the (possibly installed) module
import sys
sys.path.append(".")
import PyDynamic.misc.noise as pn

possible_inputs = [2, 0.5, 1, 0, None, -0.78, -1, -2.0]
possible_lengths = [999,1000]

def test_power_law_noise(visualize=False):

    # check function output for even/uneven N and different alphas
    for alpha in possible_inputs:

        # definitions
        N = np.random.choice(possible_lengths)
        std = 0.5

        # calculate theortic covariance
        Rxx = pn.power_law_acf(N, alpha=alpha, std=std)
        assert Rxx.shape == (N,)

        if alpha == 0.0:
            assert Rxx[0] == pytest.approx(std**2)
            assert np.mean(Rxx[1:]) == pytest.approx(0)

        # transform w into correlated nois
        w = pn.white_gaussian(N)
        w_color1 = pn.power_law_noise(w = w, std=std, alpha=alpha)
        assert w_color1.shape == (N)
        assert np.all(np.isfinite(w_color1))

        w_color2 = pn.power_law_noise(N = N, std=std, alpha=alpha)
        assert w_color2.shape == (N)
        assert np.all(np.isfinite(w_color2))

        # visualize the outcome
        if visualize:
            if alpha in [2,1,-2.0]:
                plt.figure("time_alpha=" + str(alpha))
                plt.plot(w)
                plt.plot(w_color1)

                plt.figure("psd_alpha=" + str(alpha))
                plt.psd(w_color1)
                plt.xscale("log")

                #plot.figure("Rxx_matrix_alpha=" + str(alpha))
                #plt.imshow(scipy.linalg.toeplitz(Rxx))

    if visualize:
        plt.show()

def test_noise_invalid_color():
    # check error-message for undefined color-name
    with pytest.raises(NotImplementedError) as e:
        pn.power_law_acf(N=100, color = "nocolor", )

def test_noise_user_warning():
    with pytest.raises(UserWarning) as w:
        pn.power_law_noise(N=100, color = "red", alpha = 2)
