# -*- coding: utf-8 -*-
"""
This module contains a very first draft for the utilization of the experimental
`Signal` class.

.. note:: The `Signal` class in :mod:`PyDynamic.signals` is experimental and its
signatures might change in the future or it might as well disappear completely. Let
us know if you have special needs or want to make extended use of this feature.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as dsp

from PyDynamic import rect
from PyDynamic.signals import Signal


def demonstrate_signal():
    N = 1024
    delta_t = 0.01
    t = np.arange(0, N * delta_t, delta_t)
    x = rect(t, delta_t * N // 4, delta_t * N // 4 * 3)
    ux = 0.02
    signal = Signal(t, x, Ts=delta_t, uncertainty=ux)
    b = dsp.firls(
        15,
        [0, 0.2 * signal.Fs / 2, 0.25 * signal.Fs / 2, signal.Fs / 2],
        [1, 1, 0, 0],
        nyq=signal.Fs / 2,
    )
    Ub = np.diag(b * 1e-1)
    signal.apply_filter(b, filter_uncertainty=Ub)
    signal.plot_uncertainty()
    plt.show()
    bl, al = dsp.bessel(4, 0.2)
    Ul = np.diag(np.r_[al[1:] * 1e-3, bl * 1e-2] ** 2)
    signal.apply_filter(bl, al, filter_uncertainty=Ul)
    signal.plot_uncertainty(fignr=3)
    plt.show()


if __name__ == "__main__":
    demonstrate_signal()
