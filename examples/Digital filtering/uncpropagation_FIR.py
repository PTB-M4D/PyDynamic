# -*- coding: utf-8 -*-

"""

Uncertainty propagation for a FIR lowpass filter with uncertain cut-off frequency for a rectangular signal.

.. seealso:: :mod:`uncertainty.propagate_FIR`

"""

import matplotlib.pyplot as plt
import numpy as np

from PyDynamic.misc.testsignals import rect
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter
from PyDynamic.misc.tools import col_hstack, make_semiposdef
from PyDynamic.misc.filterstuff import kaiser_lowpass
import PyDynamic.uncertainty.propagate_MonteCarlo as MC

# parameters of simulated measurement
Fs = 100e3
Ts = 1 / Fs

# nominal system parameters
fcut = 20e3
L = 100
b = kaiser_lowpass(L,fcut,Fs)[0]

# uncertain knowledge: cutoff between 19.5kHz and 20.5kHz
runs = 1000
FC = fcut + (2*np.random.rand(runs)-1)*0.5e3

B = np.zeros((runs,L+1))
for k in range(runs):
	B[k,:] = kaiser_lowpass(L,FC[k],Fs)[0]

Ub = make_semiposdef(np.cov(B,rowvar=0))

# simulate input and output signals
time = np.arange(0,499*Ts,Ts)
noise = 1e-5
x = rect(time,100*Ts,250*Ts,1.0,noise=noise)

y, Uy = FIRuncFilter(x, noise, b, Ub, blow=b)
yMC,UyMC = MC.MC(x,noise,b,[1.0],Ub,runs=1000,blow=b)

plt.figure(1); plt.cla()
plt.plot(time, x, label="input")
plt.plot(time, y, label="output")
plt.xlabel("time / au")
plt.ylabel("signal amplitude / au")
plt.legend()

plt.figure(2);plt.cla()
plt.plot(time, Uy, label="FIR formula")
plt.plot(time, np.sqrt(np.diag(UyMC)), label="Monte Carlo")
plt.xlabel("time / au")
plt.ylabel("signal uncertainty/ au")
plt.legend()
plt.show()