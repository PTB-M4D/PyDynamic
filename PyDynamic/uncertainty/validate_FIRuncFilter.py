import scipy.signal as scs
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(".")

from PyDynamic.uncertainty.propagate_filter import FIRuncFilter

# monte carlo validation
n_mc = 1000  # monte carlo runs

for nf in [10, 101]:

    # init filter
    a = [1.0]
    b = scs.firwin(nf, 0.3)
    b[-1] = 10  # make the filter asymmetric
    plt.stem(b)
    plt.show()

    for nx in [1000]:
            
        # define input signal and uncertainty of input signal
        x = np.clip(np.linspace(1,nx,nx), -10, 30)
        Ux = np.ones((nx))
        Ux[nx//2:] = 2
        Ux = 10 * Ux

        # apply filter with uncertainty
        y, Uy = FIRuncFilter(x, Ux, b, kind="diag")

        # actual monte carlo
        tmp_y = []
        for i in range(n_mc):
            x_mc = x + np.random.randn(nx) * Ux
            y_mc = scs.lfilter(b, a, x_mc)
            tmp_y.append(y_mc)

        # get distribution of results
        y_mc_mean = np.mean(tmp_y, axis=0)
        y_mc_std = np.std(tmp_y, axis=0)

        # visualize
        fig, ax = plt.subplots(nrows=1, ncols=1)

        ## plot input
        ax.plot(x, color="g", label="x")
        ax.plot(x + Ux, color="g", linestyle=":", label="x + Ux")
        ax.plot(x - Ux, color="g", linestyle=":", label="x - Ux")
        ## plot pydynamic restoration
        ax.plot(y, color="r", label="y")
        ax.plot(y + Uy, color="r", linestyle=":", label="y + Uy")
        ax.plot(y - Uy, color="r", linestyle=":", label="y - Uy")
        ## plot monte carlo results
        ax.plot(y_mc_mean, color="k", label="y_mc_mean")
        ax.plot(y_mc_mean + y_mc_std, color="k", linestyle=":", label="y_mc_mean + std")
        ax.plot(y_mc_mean - y_mc_std, color="k", linestyle=":", label="y_mc_mean - std")

        ## show plot
        ax.legend()
        plt.show()