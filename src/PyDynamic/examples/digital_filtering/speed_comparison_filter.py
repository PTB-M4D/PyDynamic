import time as tm

import matplotlib.pyplot as plt
import numpy as np
import PyDynamic.uncertainty.propagate_filter as pf
import scipy.signal as dsp
from PyDynamic.misc.testsignals import rect

for kind in ["corr", "diag"]:

    for nx in [50, 100, 500, 1000, 5000, 10000]:  # 50000, 100000, 500000
        # time
        Fs = 100e3  # sampling frequency (in Hz)
        Ts = 1 / Fs  # sampling interval length (in s)
        time = np.arange(nx) * Ts  # time values

        # init filter
        a = np.array([1.0])
        b = dsp.firwin(5, 0.3)
        Uab = 0.00001 * np.diag([1] * (a.size - 1) + [2] * b.size)
        # Uab = None

        # input signal + run methods
        sigma_noise = 1e-2
        x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=sigma_noise)

        if kind == "corr":
            c = np.exp(-np.arange(1, 10))
            Ux = sigma_noise ** 2 * c / c[0]
        else:
            Ux = sigma_noise * np.ones_like(x)

        # test all-at-once performance
        t1 = tm.time()
        y1, Uy1, _ = pf.IIRuncFilter(x, Ux, b, a, Uab=Uab, kind=kind)
        t2 = tm.time()
        y2, Uy2 = pf.FIRuncFilter(x, Ux, b, Utheta=Uab, kind=kind)
        t3 = tm.time()
        print("nx = {NX}, kind = {KIND}".format(NX=nx, KIND=kind))
        print("IIR took {0} seconds".format(t2 - t1))
        print("FIR took {0} seconds".format(t3 - t2))
        print("=" * 30)

# visualize
fig, ax = plt.subplots(nrows=1, ncols=1)

# plot input
ax.plot(x, color="g", label="x")
# plot pydynamic restoration
ax.plot(y1, color="r", label="y1")
ax.plot(y1 + Uy1, color="r", linestyle=":", label="y1 + Uy1")
ax.plot(y1 - Uy1, color="r", linestyle=":", label="y1 - Uy1")
# plot monte carlo results
ax.plot(y2, color="k", label="y2")
ax.plot(y2 + Uy2, color="k", linestyle=":", label="y2 + Uy2")
ax.plot(y2 - Uy2, color="k", linestyle=":", label="y2 + Uy2")

# show plot
ax.legend()
plt.show()
