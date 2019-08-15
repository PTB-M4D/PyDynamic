# -*- coding: utf-8 -*-
"""

This example demonstrates the use of the transformation from real-imag representation to an amplitude-phase representation and back.
It uses the corresponding PyDynamic functions from the module `uncertainty.propagate_DFT´.

The function Time2AmpPhase uses GUM_DFT and DFT2AmpPhase internally.
The function AmpPhase2Time applies the formulas as given in the corresponding publication on GUM2DFT.

"""

import numpy as np
import matplotlib.pyplot as plt
from PyDynamic.uncertainty.propagate_DFT import AmpPhase2Time, Time2AmpPhase, GUM_DFTfreq
from PyDynamic.misc.testsignals import multi_sine

# set amplitude values of multi-sine componentens (last parameter is number of components)
sine_amps = np.random.randint(1,4,10)
# set frequencies of multi-sine components
sine_freqs= np.linspace(100, 500, len(sine_amps))*2*np.pi
# define time axis
dt = 0.0001
time = np.arange(0.0, 0.2, dt)
# measurement noise standard deviation (assume white noise)
sigma_noise = 0.001
# generate test signal
testsignal = multi_sine(time, sine_amps, sine_freqs, noise=sigma_noise)

plt.figure(1, figsize=(12,6))
plt.plot(time, testsignal)
plt.xlabel("time in s")
plt.ylabel("signal amplitude in a.u.")

# uncertainty propagation from time domain to frequency domain
A,P, UAP = Time2AmpPhase(testsignal, sigma_noise**2)
f = GUM_DFTfreq(len(time), dt)

plt.figure(2, figsize=(12,6))
plt.errorbar(f, A, np.sqrt(np.diag(UAP)[:len(A)]), fmt=".-")
plt.xlabel("frequency in Hz")
plt.ylabel("DFT magnitude values in a.u.")

# uncertainty propagation from frequency domain to time domain
x, ux = AmpPhase2Time(A, P, UAP)

plt.figure(3, figsize=(12,6))
plt.subplot(211)
plt.errorbar(time, x, np.sqrt(np.diag(ux)), fmt=".-", label="after DFT and iDFT using amplitude and phase")
plt.plot(time, testsignal, label="original signal")
plt.xlabel("time in s")
plt.ylabel("signal amplitude in a.u.")
plt.legend()
plt.subplot(212)
plt.plot(time, np.sqrt(np.diag(ux)), label="uncertainty after DFT and iDFT using amplitude and phase")
plt.plot(time, np.ones_like(time)*sigma_noise, label="original uncertainty")
plt.xlabel("time in s")
plt.ylabel("uncertainty in a.u.")
plt.legend()