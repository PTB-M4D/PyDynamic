import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as dsp

import PyDynamic.uncertainty.propagate_filter as pf
from PyDynamic.misc.testsignals import rect

# define filter
wp = 0.2
ws = 0.3
gpass = 0.1
gstop = 40
# noinspection PyTupleAssignmentBalance
b, _ = dsp.iirdesign(wp, ws, gpass, gstop)
a = np.ones(1)

# simulate input and output signals
Fs = 100e3  # sampling frequency (in Hz)
Ts = 1 / Fs  # sampling interval length (in s)
nx = 200
time = np.arange(nx) * Ts  # time values

# input signal + run methods
sigma_noise = 1e-2
x = rect(time, 50 * Ts, 150 * Ts, 1.0, noise=sigma_noise)
Ux = sigma_noise * np.ones_like(x)
Uab = 0.001 * np.diag(np.arange((len(a) + len(b) - 1)))

# calculate result from IIR method
y, Uy, _ = pf.IIRuncFilter(
    x, Ux, b, a, Uab=Uab, kind="diag"
)  # state=pf.get_initial_internal_state(b, a)

# calculate fir result
y_fir, Uy_fir = pf.FIRuncFilter(x, Ux, b, Utheta=Uab, kind="diag")

# calculate monte carlo result
n_mc = 10000
tmp_y = []
for i in range(n_mc):
    x_mc = x + np.random.randn(nx) * Ux
    b_tmp = b + np.random.randn(len(b)) * np.sqrt(np.diag(Uab))
    y_mc = dsp.lfilter(b_tmp, a, x_mc)
    tmp_y.append(y_mc)
y_mc_mean = np.mean(tmp_y, axis=0)
y_mc_std = np.std(tmp_y, axis=0)


# visualize
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)

# plot input
ax1.plot(x, color="g", label="x")
ax1.plot(x + Ux, color="g", linestyle=":")
ax1.plot(x - Ux, color="g", linestyle=":")

# plot iir results
ax2.plot(y, color="r", label="iir: y")
ax2.plot(y + Uy, color="r", linestyle=":")
ax2.plot(y - Uy, color="r", linestyle=":")
# plot fir results
ax2.plot(y_fir, color="b", label="fir: y")
ax2.plot(y_fir + Uy_fir, color="b", linestyle=":")
ax2.plot(y_fir - Uy_fir, color="b", linestyle=":")
# plot monte carlo results
ax2.plot(y_mc_mean, color="k", label="mc: y")
ax2.plot(y_mc_mean + y_mc_std, color="k", linestyle=":")
ax2.plot(y_mc_mean - y_mc_std, color="k", linestyle=":")

# plot only uncertainties
ax3.plot(Uy, color="r", label="iir: unc")
ax3.plot(Uy_fir, color="b", label="fir: unc")
ax3.plot(y_mc_std, color="k", label="mc: unc")

# plot differences between all three
ax4.plot(np.abs(Uy - Uy_fir), color="c", label="iir - fir: unc")
ax4.plot(np.abs(Uy_fir - y_mc_std), color="y", label="fir - mc: unc")
ax4.plot(np.abs(y_mc_std - Uy), color="g", label="mc - iir: unc")

# prettify
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax4.set_yscale("log")

# show plot
plt.show()
