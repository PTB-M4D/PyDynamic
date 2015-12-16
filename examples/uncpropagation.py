# -*- coding: utf-8 -*-
"""

Demonstration for using the methods in the module :mod:`PyDynamic.uncpropagation`.


"""

import numpy as np
import scipy.signal as dsp


def example_FIRformula():
    """
    
    Uncertainty propagation for a FIR lowpass filter with uncertain
    cut-off frequency for a rectangular signal.
    
    .. seealso:: :mod:`PyDynamic.uncpropagation.FIR`
        
    """    
    from PyDynamic.misc.testsignals import rect
    import matplotlib.pyplot as plt
    import PyDynamic.uncpropagation.FIR as FIRunc
    from PyDynamic.misc.tools import col_hstack
    from PyDynamic.misc.filterstuff import kaiser_lowpass
    import PyDynamic.uncpropagation.MonteCarlo as MC

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
    
    y,Uy = FIRunc.FIRuncFilter(x,noise,b,Ub)    
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


def example1_IIRformula(): 
    """
    
    Uncertainty propagation for an IIR lowpass filter with uncertain
    cut-off frequency for a rectangular signal.
    
    .. seealso:: :mod:`PyDynamic.uncpropagation.IIR`
        
    """        
    
    from PyDynamic.misc.testsignals import rect
    import matplotlib.pyplot as plt
    import PyDynamic.uncpropagation.IIR as IIR    
    import PyDynamic.uncpropagation.MonteCarlo as MC
    from PyDynamic.misc.tools import col_hstack

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
        
    Uab = np.cov(AB,rowvar=0)
    
    time = np.arange(0,499*Ts,Ts)
    t0 = 100*Ts; t1 = 300*Ts
    height = 0.9
    noise = 1e-3
    x = rect(time,t0,t1,height,noise=noise)
        
    y,Uy = IIR.IIR_uncFilter(x,noise,b,a,Uab)
    yMC,UyMC = MC.MC(x,noise,b,a,Uab,runs=10000)
    
    plt.figure(1);plt.cla()
    plt.plot(time,col_hstack([x,y]))   
    plt.legend(('input','output'))
    
    plt.figure(2);plt.cla()
    plt.plot(time,col_hstack([Uy,UyMC]))
    plt.title('uncertainty of filter output')
    plt.legend(('IIR formula', 'Monte Carlo'))
    
    plt.show()
    
    
    
def example2_IIRformula(): 
    """
    
    Uncertainty propagation for an IIR bandpass filter with uncertain
    coefficients for a rectangular signal
    
    .. seealso:: :mod:`PyDynamic.uncpropagation.IIR`
        
    """        
    
    from PyDynamic.misc.testsignals import rect
    import matplotlib.pyplot as plt
    import PyDynamic.uncpropagation.IIR as IIR    
    import PyDynamic.uncpropagation.MonteCarlo as MC
    from PyDynamic.misc.tools import col_hstack

    # parameters of simulated measurement
    Fs = 100e3
    Ts = 1.0/Fs
    
    # nominal system parameter    
    fcut = np.array([8e3, 20e3])
    L = 6    
    b,a = dsp.butter(L,2*fcut/Fs,btype='bandpass')
    
    Uab = 1e-12*np.diag(np.hstack((a[1:],b))**2)
    
    time = np.arange(0,499*Ts,Ts)
    t0 = 100*Ts; t1 = 300*Ts
    height = 1.0
    noise = 1e-3
    x = rect(time,t0,t1,height,noise=noise)
        
    y,Uy = IIR.IIR_uncFilter(x,noise,b,a,Uab)
    yMC,UyMC = MC.SMC(x,noise,b,a,Uab,runs=1000)
    
    plt.figure(1);plt.cla()
    plt.plot(time,col_hstack([x,y]))   
    plt.legend(('input','output'))
    
    plt.figure(2);plt.cla()
    plt.plot(time,col_hstack([Uy,UyMC]))
    plt.title('uncertainty of filter output')
    plt.legend(('IIR formula', 'Monte Carlo'))
    
    plt.show()

