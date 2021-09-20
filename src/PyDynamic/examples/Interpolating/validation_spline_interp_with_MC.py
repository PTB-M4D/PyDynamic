# -*- coding: utf-8 -*-
"""
    Interpolate a non-equidistant sine signal using cubic / bspline
    method with uncertainty propagation. 
    Comparing the resulting uncertainties to a Monte-Carlo experiment
    yields good overlap. 
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PyDynamic.uncertainty.interpolate import interp1d_unc

# base signal
N = 10
t = np.cumsum(0.5 * (1 + np.random.random(N)))
x = np.sin(t) # + 0.01*np.random.randn(N)
#ut = np.full_like(t, 0.05)
ux = np.full_like(x, 0.2)

# interpolate with PyDynamic
ti = np.linspace(np.min(t), np.max(t), 60)
#uti = np.full_like(ti, 0.05)
ti, xi, uxi = interp1d_unc(ti, t, x, ux, kind="cubic")

# interpolate with Monte Carlo
X_mc = []
for i in range(2000):
    interp_x = interp1d(t, x + ux * np.random.randn(len(x)), kind="cubic")
    xm = interp_x(ti)
    X_mc.append(xm)
x_mc = np.mean(X_mc, axis=0)
ux_mc = np.std(X_mc, axis=0)

# visualize
# interpolated signal
plt.plot(ti, xi, '-or', label="interpolation PyDynamic")
plt.fill_between(ti, xi + uxi, xi - uxi, color="r", alpha=0.3)
# interpolated signal
plt.plot(ti, x_mc, '-ok', label="interpolation Monte Carlo")
plt.fill_between(ti, x_mc + ux_mc, x_mc - ux_mc, color="k", alpha=0.3)
# original signal
plt.plot(t, x, '-ob', label="original")
plt.fill_between(t, x + ux, x - ux, color="b", alpha=0.3)

plt.legend()
plt.show()
