"""Small script to validate DWT methods with Monte-Carlo simulations"""

import matplotlib.pyplot as plt
import numpy as np
import pywt

from PyDynamic.uncertainty.propagate_DWT import dwt, filter_design, inv_dwt

# monte carlo validation
n_mc = 1000  # monte carlo runs

for filter_name in ["db9"]:

    for nx in [101]:

        # define input signal and uncertainty of input signal
        x = np.clip(np.linspace(1, nx, nx), -10, 30)
        Ux = np.ones(nx)
        Ux[nx // 2 :] = 2
        Ux = 1.0 * Ux

        ld, hd, lr, hr = filter_design(filter_name)

        # single decomposition with uncertainty
        c_approx, U_approx, c_detail, U_detail, _ = dwt(x, Ux, ld, hd)

        # single reconstruction with uncertainty
        xr, Uxr, _ = inv_dwt(c_approx, U_approx, c_detail, U_detail, lr, hr)

        # actual monte carlo

        # MC: decomposition
        tmp_c = []
        for i in range(n_mc):
            x_mc = x + np.random.randn(nx) * Ux
            c_approx_mc, c_detail_mc = pywt.dwt(x_mc, filter_name, mode="constant")
            tmp_c.append(c_approx_mc)

        # MC: reconstruction
        tmp_x = []
        for i in range(n_mc):
            c_approx_mc = c_approx + np.random.randn(*c_approx.shape) * U_approx
            c_detail_mc = c_detail + np.random.randn(*c_detail.shape) * U_detail
            xr_mc = pywt.idwt(c_approx_mc, c_detail_mc, filter_name, mode="constant")
            tmp_x.append(xr_mc)

        # get distribution of results
        c_mc_mean = np.mean(tmp_c, axis=0)
        c_mc_std = np.std(tmp_c, axis=0)
        xr_mc_mean = np.mean(tmp_x, axis=0)
        xr_mc_std = np.std(tmp_x, axis=0)

        # visualize
        fig, ax = plt.subplots(nrows=2, ncols=1)

        # plot coefficients
        # plot pydynamic dwt results
        ax[0].plot(c_approx, color="r", label="cA")
        ax[0].plot(c_approx + U_approx, color="r", linestyle=":", label="cA + unc")
        ax[0].plot(c_approx - U_approx, color="r", linestyle=":", label="cA - unc")
        # plot monte carlo results
        ax[0].plot(c_mc_mean, color="k", label="cA_mc mean")
        ax[0].plot(c_mc_mean + c_mc_std, color="k", linestyle=":", label="cA_mc + std")
        ax[0].plot(c_mc_mean - c_mc_std, color="k", linestyle=":", label="cA_mc - std")

        # plot time series
        # plot input
        ax[1].plot(x, color="c", label="x")
        ax[1].plot(x + Ux, color="c", linestyle=":", label="x + Ux")
        ax[1].plot(x - Ux, color="c", linestyle=":", label="x - Ux")
        # plot pydynamic restoration
        ax[1].plot(xr, color="r", label="xr")
        ax[1].plot(xr + Uxr, color="r", linestyle=":", label="xr + Uxr")
        ax[1].plot(xr - Uxr, color="r", linestyle=":", label="xr - Uxr")
        # plot monte carlo results
        ax[1].plot(xr_mc_mean, color="k", label="x_mc_mean")
        ax[1].plot(xr_mc_mean + xr_mc_std, color="k", linestyle=":", label="x_mc + std")
        ax[1].plot(xr_mc_mean - xr_mc_std, color="k", linestyle=":", label="x_mc - std")

        # show plot
        ax[0].legend()
        ax[1].legend()
        plt.show()
