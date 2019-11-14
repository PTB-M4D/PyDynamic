"""
Perform test for uncertainty.propagate_DWT
"""

import matplotlib.pyplot as plt
import numpy as np

from PyDynamic.uncertainty.propagate_DWT import dwt, wave_dec, idwt, wave_rec, filter_design


def test_filter_design():

    ld, hd, lr, hr = filter_design("db3")
    print(ld, hd, lr, hr)

    ld, hd, lr, hr = filter_design("rbio3.3")
    print(ld, hd, lr, hr)


def test_dwt():

    nx = np.random.choice([20,21])
    x = np.random.randn(nx)
    Ux = 0.1 * (1 + np.random.random(nx))

    nf = np.random.choice([10,11])
    l = np.random.randn(nf)
    h = np.random.randn(nf)

    y1, Uy1, y2, Uy2 = dwt(x, Ux, l, h, kind="corr")

    # all output has same length
    assert y1.size == y2.size
    assert y1.size == Uy1.size
    assert Uy1.size == Uy2.size

    # output is half the length of input
    assert (x.size + 1) // 2 == y1.size


def test_decomposition():

    t = np.linspace(0,10,100)
    x = np.sin(t) + t + np.random.randn(*t.shape)

    nf = 4
    l = np.ones((nf,)) / nf
    h = l

    results = wave_dec(x, x, l, h, max_depth=-1)

    assert isinstance(results, list)


def test_idwt():
    pass


def test_reconstruction():
    pass


def test_identity():

    nx = np.random.choice([200,201])
    x = np.random.randn(nx)
    Ux = 0.1 * (1 + np.random.random(nx))

    ld, hd, lr, hr = filter_design("db3")

    # single decomposition
    y_approx, U_approx, y_detail, U_detail = dwt(x, Ux, ld, hd, kind="diag")

    # single reconstruction
    xr, Uxr = idwt(y_approx, U_approx, y_detail, U_detail, lr, hr, kind="diag")

    plt.plot(x)
    plt.plot(xr)
    plt.show()
    print(x)
    print(xr)