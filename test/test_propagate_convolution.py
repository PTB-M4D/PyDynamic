"""
Test PyDynamic.uncertainty.propagate_convolve 
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz

from PyDynamic.uncertainty.propagate_convolution import convolve_unc


def test_convolution():
    # check for even/uneven and longer/equal/shorter n1 + n2 
    for n1 in [10, 15, 20]:
        for n2 in [10, 15, 20]:

            # define signals
            x1 = np.random.randn(n1)
            u1 = 0.01 * (1 + np.random.rand(n1))

            x2 = np.random.randn(n2)
            u2 = 0.01 * (1 + np.random.rand(n2))

            # test all modes of numpy convolve
            for mode in ["full", "valid", "same"]:

                # calculate the convolution of x1 and x2
                y, Uy = convolve_unc(x1, u1, x2, u2, mode=mode)
                y_numpy = np.convolve(x1, x2, mode=mode)

                # compare results
                assert len(y) == len(Uy)
                assert len(y) == len(y_numpy)
                assert np.allclose(y, y_numpy)
                
                # visualize
                # print(n1, n2, mode)
                # plt.plot(y, label="PyDynamic")
                # plt.plot(y_numpy, label="NumPy")
                # plt.legend()
                # plt.show()