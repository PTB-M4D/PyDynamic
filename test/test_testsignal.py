# -*- coding: utf-8 -*-
"""
Perform tests on identification sub-packages.

"""

import numpy as np
import pytest

# import PyDynamic from local code, not from the (possibly installed) module
import sys
sys.path.append(".")
from PyDynamic.misc.testsignals import corr_noise



def test_corrNoiseBeta():
    import matplotlib.pyplot as plt

    lengths = [99, 100]

    # check function output for even/uneven N and different betas
    for beta in [2, 0.5, 0, None, -0.78, -1, -2.0]:

        # definitions
        N     = np.random.choice(lengths)
        sigma = 0.5
        w     = np.random.normal(loc = 0, scale = sigma, size=N)

        # instantiate correlated noise class
        cn = corr_noise(w, sigma)

        # calculate theortic covariance
        Rxx = cn.theoretic_covariance_colored_noise(beta=beta)
        assert Rxx.shape == (2*N,)
        if beta == 0.0:
            assert Rxx[0] == pytest.approx(sigma**2)
            assert np.mean(Rxx[1:]) == pytest.approx(0)

        # transform w into correlated noise
        w_color = cn.colored_noise(beta=beta)
        assert w_color.shape == (N)
        assert np.all(np.isfinite(w_color))

        # visualize the outcome
        #if beta == 2:
        #    print(Rxx)
        #    plt.imshow(Rxx)
        #    plt.show()
        #    print(np.std(w))
        #    print(np.std(w_color))
        #    plt.plot(w)
        #    plt.plot(w_color)
        #    plt.show()

# add test to compare old and new implementations?

def test_corrNoiseInvalidColor():
    # check error-message for undefined color-name
    cn = corr_noise(None, 1.0)
    with pytest.raises(NotImplementedError) as e:
        cn.theoretic_covariance_colored_noise(100, color = "nocolor")

def test_corrNoiseUserWarning():
    cn = corr_noise(None, 1.0)
    with pytest.raises(UserWarning) as w:
        cn.theoretic_covariance_colored_noise(100, color = "red", beta = 2)
