# -*- coding: utf-8 -*-

"""

Uncertainty propagation for an IIR lowpass filter with uncertain
cut-off frequency for a rectangular signal.

.. seealso:: :mod:`uncertainty.propagate_IIR`

"""

import matplotlib.pyplot as plt
import scipy.signal as dsp
import numpy as np

from PyDynamic.misc.testsignals import rect
from PyDynamic.uncertainty.propagate_filter import IIR_uncFilter
import PyDynamic.uncertainty.propagate_MonteCarlo as MC
from PyDynamic.misc.tools import make_semiposdef

# parameters of simulated measurement
Fs = 100e3
Ts = 1.0/Fs

# nominal system parameter
fcut = 20e3
L = 6
b,a = dsp.butter(L,2*fcut/Fs,btype='lowpass')

# uncertain knowledge: fcut between 19.8kHz and 20.2kHz
runs = 1000
FC = fcut + (2*np.random.rand(runs)-1)*0.2e3
AB = np.zeros((runs,len(b)+len(a)-1))

for k in range(runs):
	bb,aa = dsp.butter(L,2*FC[k]/Fs,btype='lowpass')
	AB[k,:] = np.hstack((aa[1:],bb))

Uab = make_semiposdef(np.cov(AB,rowvar=0))

time = np.arange(0,499*Ts,Ts)
t0 = 100*Ts; t1 = 300*Ts
height = 0.9
noise = 1e-3
x = rect(time,t0,t1,height,noise=noise)

y,Uy = IIR_uncFilter(x, noise, b, a, Uab)
yMC,UyMC = MC.SMC(x,noise,b,a,Uab,runs=10000)

plt.figure(1);plt.cla()
plt.plot(time*1e3, x, label="input signal")
plt.plot(time*1e3, y, label="output signal")
plt.xlabel('time / ms',fontsize=22)
plt.ylabel('signal amplitude / au',fontsize=22)
plt.tick_params(which="both",labelsize=16)

plt.figure(2);plt.cla()
plt.plot(time*1e3, Uy, label='IIR formula')
plt.plot(time*1e3, UyMC, label='Monte Carlo')
# plt.title('uncertainty of filter output')
plt.legend()
plt.xlabel('time / ms',fontsize=22)
plt.ylabel('uncertainty / au',fontsize=22)
plt.tick_params(which='both',labelsize=16)

plt.show()


