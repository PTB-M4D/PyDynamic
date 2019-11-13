"""
Perform test for uncertainty.propagate_DWT
"""

import matplotlib.pyplot as plt
import numpy as np

from PyDynamic.uncertainty.propagate_DWT import DWT, wavelet_block, DWT_filter_design


def test_DWT_filter_design():

    ld, hd, lr, hr = DWT_filter_design("db3")
    print(ld, hd, lr, hr)

    ld, hd, lr, hr = DWT_filter_design("rbio3.3")
    print(ld, hd, lr, hr)


def test_wavelet_block():

    nx = np.random.choice([20,21])
    x = np.random.randn(nx)
    Ux = 0.1 * (1 + np.random.random(nx))

    nf = np.random.choice([10,11])
    g = np.random.randn(nf)
    h = np.random.randn(nf)

    y1, Uy1, y2, Uy2 = wavelet_block(x, Ux, g, h, kind="corr")

    # all output has same length
    assert y1.size == y2.size
    assert y1.size == Uy1.size
    assert Uy1.size == Uy2.size

    # output is half the length of input
    assert (x.size + 1) // 2 == y1.size


def test_DWT():

    t = np.linspace(0,10,100)
    x = np.sin(t) + t + np.random.randn(*t.shape)

    nf = 4
    g = np.ones((nf,)) / nf
    h = g


    results = DWT(x, x, g, h, max_depth=-1)

    assert isinstance(results, list)