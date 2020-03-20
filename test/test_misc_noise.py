"""
Perform tests on noise methods.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pytest

# import PyDynamic from local code, not from the (possibly installed) module
import sys
sys.path.append(".")
import PyDynamic.misc.noise as pn

possible_inputs = [2, 0.5, 1, 0, -0.78, -1, -2.0, "blue", "red", "white"]
possible_lengths = [99,100]

def test_get_alpha():
    # typical behavior
    assert pn.get_alpha("red") == -2
    assert pn.get_alpha(2) == 2.0
    assert pn.get_alpha(0.4) == 0.4

    # check error-message for undefined colorname
    with pytest.raises(NotImplementedError) as e:
        pn.get_alpha("nocolor")

def test_white_gaussian():
    w = pn.white_gaussian(10)
    assert len(w) == 10

def test_power_law_noise(visualize=False):

    # check function output for even/uneven N and different alphas
    for color_value in possible_inputs:

        # definitions
        N = np.random.choice(possible_lengths)
        std = 0.5

        # transform w into correlated noise
        w = pn.white_gaussian(N)
        w_color1 = pn.power_law_noise(w = w, std=std, color_value=color_value)
        assert w_color1.shape == (N)
        assert np.all(np.isfinite(w_color1))

        # produce noise of length N
        w_color2 = pn.power_law_noise(N = N, std=std, color_value=color_value)
        assert w_color2.shape == (N)
        assert np.all(np.isfinite(w_color2))

        # visualize the outcome
        if visualize:
            if color_value in [2,1,-2.0]:
                plt.figure("time_alpha=" + str(color_value))
                plt.plot(w)
                plt.plot(w_color1)

                plt.figure("psd_alpha=" + str(color_value))
                plt.psd(w_color1)
                plt.xscale("log")

                #plot.figure("Rxx_matrix_alpha=" + str(color_value))
                #plt.imshow(scipy.linalg.toeplitz(Rxx))

    if visualize:
        plt.show()

def test_power_law_acf():

    # check function output for even/uneven N and different alphas
    for color_value in possible_inputs:

        # definitions
        N = np.random.choice(possible_lengths)
        std = 0.5

        # calculate theoretic covariance
        Rxx = pn.power_law_acf(N, color_value=color_value, std=std)
        assert Rxx.shape == (N,)

        if color_value == 0.0:
            assert Rxx[0] == pytest.approx(std**2)
            assert np.mean(Rxx[1:]) == pytest.approx(0)

def test_ARMA():
    
    # check default parameters (white noise)
    w = pn.ARMA(100)
    assert w.shape == (100,)
    assert np.std(w) == pytest.approx(1.0, abs=0.2)
    assert np.mean(w) == pytest.approx(0.0, abs=0.2)

    # check float, list and numpy.arrays as input values
    phi_list = [2, [2], np.array([1,2])]
    theta_list = [1, [1], np.array([1,2])]

    for theta, phi in zip(theta_list, phi_list):

        # definitions
        N = np.random.choice(possible_lengths)
        std = 0.5

        # calculate ARMA processes
        w = pn.ARMA(100, phi=phi, theta=theta, std=std)
        assert w.shape == (100,)