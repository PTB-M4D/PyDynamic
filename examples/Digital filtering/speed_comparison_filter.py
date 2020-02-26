import sys
sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scs
import time as tm
from collections import deque

from PyDynamic.misc.testsignals import rect
import PyDynamic.uncertainty.propagate_filter as pf


# define some buffer to fill and consume
class Buffer():

    def __init__(self, maxlen=1000):
        self.timestamps = deque(maxlen=maxlen)
        self.values = deque(maxlen=maxlen)
        self.uncertainties = deque(maxlen=maxlen)
    
    def append(self, time=0.0, value=0.0, uncertainty=0.0):
        self.timestamps.append(time)
        self.values.append(value)
        self.uncertainties.append(uncertainty)

    def popleft(self):
        t = self.timestamps.popleft()
        v = self.values.popleft()
        u = self.uncertainties.popleft()

        return t, v, u


#for nx in [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]:
for nx in [5000]:
    # time
    Fs = 100e3        # sampling frequency (in Hz)
    Ts = 1 / Fs       # sampling interval length (in s)
    time  = np.arange(nx)*Ts                     # time values

    # init filter
    a = np.array([1.0])
    b = scs.firwin(5, 0.3)
    Uab = 0.00001*np.diag([1]*(a.size-1) + [2]*b.size)
    #Uab = None

    # input signal + run methods
    sigma_noise = 1e-2
    x = rect(time,100*Ts,250*Ts,1.0,noise=sigma_noise)
    #Ux = sigma_noise * np.ones_like(x)
    c = np.exp(-np.arange(1,10))
    Ux = sigma_noise * c / c[0]

    # test all-at-once performance
    t1 = tm.time()
    y1, Uy1, _ = pf.IIRuncFilter(x, Ux, b, a, Uab=Uab, kind="corr")
    t2 = tm.time()
    y2, Uy2 = pf.FIRuncFilter(x, Ux, b, Utheta=Uab, kind="corr")
    t3 = tm.time()
    print("nx = {0}".format(nx))
    print("IIR took {0} seconds".format(t2-t1))
    print("FIR took {0} seconds".format(t3-t2))
    print("="*20)

# visualize
fig, ax = plt.subplots(nrows=1, ncols=1)

## plot input
ax.plot(x, color="g", label="x")
#ax.plot(x + Ux, color="g", linestyle=":", label="x + Ux")
#ax.plot(x - Ux, color="g", linestyle=":", label="x - Ux")
## plot pydynamic restoration
ax.plot(y1, color="r", label="y1")
ax.plot(y1 + Uy1, color="r", linestyle=":", label="y1 + Uy1")
ax.plot(y1 - Uy1, color="r", linestyle=":", label="y1 - Uy1")
## plot monte carlo results
ax.plot(y2, color="k", label="y2")
ax.plot(y2 + Uy2, color="k", linestyle=":", label="y2 + Uy2")
ax.plot(y2 - Uy2, color="k", linestyle=":", label="y2 + Uy2")

## show plot
ax.legend()
plt.show()








# test realtime performance
