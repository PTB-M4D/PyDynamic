import sys
sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scs
import time as time_measure

from PyDynamic.misc.testsignals import rect
import PyDynamic.uncertainty.propagate_filter as pf


# define filter
wp = 0.2
ws = 0.3
gpass = 0.1
gstop = 40
b, a = scs.iirdesign(wp, ws, gpass, gstop)

# simulate input and output signals
Fs = 100e3        # sampling frequency (in Hz)
Ts = 1 / Fs       # sampling interval length (in s)
nx = 500
time  = np.arange(nx)*Ts                     # time values

# input signal + run methods
sigma_noise = 1e-2
x = rect(time,100*Ts,250*Ts,1.0,noise=sigma_noise)
Ux = sigma_noise * np.ones_like(x)
Uab = np.diag(np.zeros((len(a) + len(b) -1)))
Uab[2,2] = 0.000001

y, Uy, _ = pf.IIRuncFilter(x, Ux, b, a, Uab=Uab, kind="diag")
#y, Uy = pf.IIRuncFilter(x, Ux, b, a, Uab=Uab, kind="diag", state=pf.get_initial_internal_state(b, a))

# actual monte carlo
n_mc = 1000
tmp_y = []
for i in range(n_mc):
    x_mc = x + np.random.randn(nx) * Ux
    a_tmp = np.copy(a)
    a_tmp[2] = a[2] + np.sqrt(Uab[2,2]) * np.random.randn()
    y_mc = scs.lfilter(b, a_tmp, x_mc)
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