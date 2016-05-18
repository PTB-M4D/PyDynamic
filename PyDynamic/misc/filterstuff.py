# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Sascha Eichstaedt (sascha.eichstaedt@ptb.de)

A collection of methods which are related to filter design.

"""

def db(vals):
    # Calculation of decibel values :math:`20\log_{10}(x)` for a vector of values    
    from numpy import log10,abs
    dbvals = 20*log10(abs(vals))
    return dbvals
    
def ua(vals):
    # Calculation of unwrapped angle of complex values    
    from numpy import unwrap,angle
    return unwrap(angle(vals))

    
def grpdelay(b,a,Fs,nfft=512):
    """Calculation of the group deleay of a digital filter
   
    Parameters:
        b:    ndarray
              IIR filter numerator coefficients
        a:    ndarray
              IIR filter denominator coefficients
        Fs:   float
              sampling frequency of the filter
        nfft: int
              number of FFT bins

    Returns:
        group delay: ndarray
                     group delay values
        frequencies: ndarray
                     frequencies at which the group delay is calculated    

    References
    * Smith, online book [Smith]_

    """
    from numpy import convolve,arange,nonzero,real
    from numpy.fft import fft
            
    Na = len(a)-1
    Nb = len(b)-1
    
    c = convolve(b,a[::-1]) # c(z) = b(z)*a(1/z)*z^(-oa)
    cr = c*arange(Na+Nb+1)  # derivative of c wrt 1/z
    num = fft(cr,2*nfft);
    den = fft(c,2*nfft);
    tol = 1e-12
    
    polebins = nonzero(abs(den)<tol) 
    for p in polebins:        
        num[p] = 0.0
        den[p] = 1.0
        
    gd = real(num/den) - Na
    
    f = arange(0.0,2*nfft-1)/(2*nfft)*Fs
    f = f[:nfft+1]
    gd = gd[:len(f)]
    return gd,f
    

def mapinside(a):
    """Maps the roots of polynomial with coefficients a inside the unit circle
    
    Parameters:
        a: ndarray
           polynomial coefficients    
    Returns:
        a: ndarray
           polynomial coefficients with all roots inside or on the unit circle
    """
    from numpy import roots,conj,poly,nonzero
    v = roots(a)
    inds = nonzero(abs(v)>1)
    v[inds] = 1/conj(v[inds])
    return poly(v)    
    

def kaiser_lowpass(L,fcut,Fs,beta=8.0):
    """
    Design of a FIR lowpass filter using the window technique with a Kaiser window.
    
    Parameters:
        L: int
           filter order (window length)
        fcut: float
              desired cut-off frequency
        Fs: float
            sampling frequency
        beta: float
              scaling parameter for the Kaiser window
    Returns:
        blow: ndarray
              FIR filter coefficients
        shift: int
               delay of the filter (in samples)
    
    """
    from scipy.signal import firwin    
    from numpy import mod
    if mod(L,2)==0:
        L = L+1
    blow = firwin(L,2*fcut/Fs,window=('kaiser',beta))
    shift = L/2
    return blow, shift
    
    
    
def isstable(b,a,ftype='digital'):
    """Determine whether IIR filter (b,a) is stable
    
    Parameters:
        b:      ndarray
                filter numerator coefficients
        a:      ndarray
                filter denominator coefficients
        ftype:  string
                type of filter (`digital` or `analog`)
    Returns:
        stable: boolean
                flag whether filter is stable or not
                
    Note
    The test for analog filters is not implemented yet.
    
    """    
    from numpy import roots, any, abs
    
    if ftype=='digital':
        v = roots(a)
        if any(abs(v)>1.0):
            return False
        else:
            return True
    elif ftype=='analog':
        raise NotImplemented
        
        
    
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    
    Source obtained from scipy cookbook (online), downloaded 2013-09-13    

    Parameters
        y: ndarray, shape (N,)
           the values of the time history of the signal
        window_size: int
                     the length of the window. Must be an odd integer number
        order: int
               the order of the polynomial used in the filtering. Must be less then `window_size` - 1.
        deriv: int
               the order of the derivative to compute (default = 0 means only smoothing)
    
    Returns
         ys: ndarray, shape (N,)
             the smoothed signal (or it's n-th derivative).
    
    Notes
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    
    Example 
    .. code-block:: python
    
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
    
    References
    * Savitzky et al. [Savitzky]_
    * Numerical Recipes [NumRec]_
       
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')    
    
