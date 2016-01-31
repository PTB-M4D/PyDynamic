# -*- coding: utf-8 -*-

"""

Uncertainty propagation for a FIR lowpass filter with uncertain
cut-off frequency for a rectangular signal.

.. seealso:: :mod:`PyDynamic.uncpropagation.FIR`

"""

# if run as script, add parent path for relative importing
if __name__ == '__main__' and __package__ is None:
	from os import sys, path
	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np

from misc.testsignals import rect
from uncertainty.propagate_FIR import FIRuncFilter
from misc.tools import col_hstack
from misc.filterstuff import kaiser_lowpass
import uncertainty.propagate_MonteCarlo as MC

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

Ub = np.cov(B,rowvar=0)

# simulate input and output signals
time = np.arange(0,499*Ts,Ts)
noise = 1e-3
x = rect(time,100*Ts,250*Ts,1.0,noise=noise)

y,Uy = FIRuncFilter(x,noise,b,Ub)
yMC,UyMC = MC.MC(x,noise,b,[1.0],Ub,runs=10000)
yMC2,UyMC2 = MC.SMC(x,noise,b,[1.0],Ub,runs=10000)

plt.figure(1); plt.cla()
plt.plot(time,col_hstack([x,y]))
plt.legend(('input','output'))

plt.figure(3);plt.cla()
plt.plot(time,col_hstack([Uy,UyMC,UyMC2]))
plt.title('Uncertainty of filter output signal')
plt.legend(('FIR formula','Monte Carlo','Sequential Monte Carlo'))

plt.show()