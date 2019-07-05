# -*- coding: utf-8 -*-
"""
Perform tests on identification sub-packages.

"""

import numpy as np

# import PyDynamic from local code, not from the (possibly installed) module
import sys
sys.path.append(".")
from PyDynamic.misc.testsignals import corr_noise



def test_colored_noise():
    import matplotlib.pyplot as plt

    lengths = [99, 100]

    # check function output for even/uneven N and different betas
    for beta in [-2, -0.5, 0, None, 0.78, 1, 2.0]:

        # definitions
        N     = np.random.choice(lengths)
        sigma = 0.5
        w     = np.random.normal(loc = 0, scale = sigma, size=N)

        # instantiate correlated noise class
        cn = corr_noise(w, sigma)

        # calculate theortic covariance
        Rxx = cn.theoretic_covariance_colored_noise(beta=beta)
        assert Rxx.shape == (N,)
        
        # transform w into correlated noise
        w_color = cn.colored_noise(beta=beta)
        assert w_color.shape == (N)
        assert np.all(np.isfinite(w_color))

        # visualize the outcome
        if beta == -2:
            print(Rxx)
            plt.imshow(Rxx)
            plt.show()
            print(np.std(w))
            print(np.std(w_color))
            plt.plot(w)
            plt.plot(w_color)
            plt.show()


    # check error-message for undefined color-name
    try:
        Rxx = cn.theoretic_covariance_colored_noise(N, color = "NOCOLOR")
    except NotImplementedError as e:
        print(e)
        assert True
    else: 
        assert False


test_colored_noise()