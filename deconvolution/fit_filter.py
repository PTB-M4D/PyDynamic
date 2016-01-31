"""
.. module:: LSIIR

.. moduleauthor:: Sascha Eichstaedt (sascha.eichstaedt@ptb.de)

This module contains methods to fit a digital filter to a given frequency response
such that the resulting filter acts as deconvolution/inverse filter for the corresponding
measurement system. For an example see :mod:`ADM.examples.LSIIR_deconvolution` 
Methods in this package are

LSFIR(H,N,tau,f,Fs,Wt=None)
LSIIR(Hvals,Nb,Na,f,Fs,tau,justFit=False,verbose=True)
LSFIR_unc(H,UH,N,tau,f,Fs,wt=None,verbose=True)
LSFIR_uncMC(H,UH,N,tau,f,Fs,wt=None,verbose=True)
LSIIR_unc(H,UH,Nb,Na,f,Fs,tau=0)
FreqResp2np.realnp.imag(Abs,Phase,Unc,MCruns=1e4)

"""

import numpy as np

if __name__=="deconvolution.fit_filter":
    from misc.filterstuff import grpdelay
else:
    from ..misc.filterstuff import grpdelay


def LSFIR(H,N,tau,f,Fs,Wt=None):
    """
    Least-squares fit of a digital FIR filter to the reciprocal of a given frequency response.
    
    
    :param H: frequency response values
    :param N: FIR filter order
    :param tau: delay of filter
    :param f: frequencies
    :param Fs: sampling frequency of digital filter
    :param Wt: (optional) vector of weights 
    
    :type H: ndarray
    :type N: int
    :type tau: int
    :type f: ndarray
    :type Fs: float
    
    :returns: filter coefficients bFIR (ndarray) of shape (N,)
    
    """
    import numpy as np

    print "\nLeast-squares fit of an order %d digital FIR filter to the" % N
    print "reciprocal of a frequency response given by %d values.\n" % len(H)

    H = H[:,np.np.newaxis]

    w = 2*np.np.pi*f/Fs    
    w = w[:,np.np.newaxis]
        
    ords = np.np.arange(N+1)[:,np.np.newaxis]
    ords = ords.T
    
    E = np.np.exp(-1j*np.np.dot(w,ords))
          
    if not Wt == None:
        if len(np.shape(Wt))==2: # is matrix
            weights = np.np.diag(Wt)
        else:
            weights = np.eye(len(f))*Wt
        X = np.np.vstack([np.np.real(np.np.dot(weights,E)), np.np.imag(np.np.dot(weights,E))])
    else:
        X = np.np.vstack([np.np.real(E), np.np.imag(E)])

    H = H*np.np.exp(1j*w*tau)
    iRI = np.np.vstack([np.np.real(1.0/H), np.np.imag(1.0/H)])
    
    bFIR, res = np.linalg.lstsq(X,iRI)[:2]

    if not inp.sinstance(res,np.ndarray):
        print "Calculation of FIR filter coefficients finished with residual norm %e" % res

    return np.reshape(bFIR,(N+1,))


def mapinside(a):
    from numpy import roots,conj,poly,nonzero
    v = roots(a)
    inds = nonzero(abs(v)>1)
    v[inds] = 1/conj(v[inds])
    return poly(v)
    

def LSIIR(Hvals,Nb,Na,f,Fs,tau,justFit=False,verbose=True):
    """
    Least-squares fit of a digital IIR filter to the reciprocal of a given set 
    of frequency response values unp.sing the equation-error method and stabilization 
    by pole mapnp.ping.
    
    :param Hvals: frequency response values.
    :param Nb: order of IIR numerator polynomial.
    :param Na: order of IIR denominator polynomial.
    :param f: frequencies.
    :param Fs: sampling frequency for digital IIR filter.
    :param tau: initial estimate of time delay for filter stabilization.
    :param justFit: if True then no stabilization is carried out.
    :type justFit: bool.
    :type Hvals: ndarray
    :type Nb: int
    :type Na: int
    :type f: ndarray
    :type Fs: float
    :type tau: int 
    
    :returns: ndarray b, a -- IIR filter coefficients, int tau -- time delay (in samples)
     
    .. seealso:: :mod:`ADM.examples.deconvolution`
    """
    from numpy import conj,count_nonzero,roots,ceil,median
    from numpy.linalg import lstsq

    if verbose:
        print "\nLeast-squares fit of an order %d digital IIR filter to the" % max(Nb,Na)
        print "reciprocal of a frequency response given by %d values.\n" % len(Hvals)
  
    w = 2*np.pi*f/Fs
    Ns= np.arange(0,max(Nb,Na)+1)[:,np.newaxis]
    E = np.exp(-1j*np.dot(w[:,np.newaxis],Ns.T))
    
    def fitIIR(Hvals,tau,E,Na,Nb):
        Ea= E[:,1:Na+1]
        Eb= E[:,:Nb+1]
        Htau = np.exp(-1j*w*tau)*Hvals**(-1)
        HEa = np.dot(np.diag(Htau),Ea)
        D   = np.hstack((HEa,-Eb))
        Tmp1= np.real(np.dot(conj(D.T),D))
        Tmp2= np.real(np.dot(conj(D.T),-Htau))
        ab = lstsq(Tmp1,Tmp2)[0]
        ai = np.hstack((1.0,ab[:Na]))
        bi = ab[Na:]
        return bi,ai
        
    bi,ai = fitIIR(Hvals,tau,E,Na,Nb)
    
    if justFit:
        return bi,ai
            
    if count_nonzero(abs(roots(ai))>1)>0:
        stable = False
    else:
        stable = True
    
    maxiter = 50
    
    astab = mapinside(ai)
    run = 1
    
    while stable!=True and run < maxiter:
        g1 = grpdelay(bi,ai,Fs)[0]
        g2 = grpdelay(bi,astab,Fs)[0]
        tau = ceil(tau + median(g2-g1))
        
        bi,ai = fitIIR(Hvals,tau,E,Na,Nb)
        if count_nonzero(abs(roots(ai))>1)>0:
            astab = mapinside(ai)
        else:
            stable = True
        run = run + 1
        
    if count_nonzero(abs(roots(ai))>1)>0 and verbose:
        print "Caution: The algorithm did NOT result in a stable IIR filter!"
        print "Maybe try again with a higher value of tau0 or a higher filter order?"
        
    if verbose:
        print "Least squares fit finished after %d iterations (tau=%d).\n" % (run,tau)
        
    return bi,ai,int(tau)
    
# TODO Implement with non-trivial weighting
def LSFIR_unc(H,UH,N,tau,f,Fs,wt=None,verbose=True,returnHi=False,trunc_svd_tol=None):
    """
    Least-squares fit of a digital FIR filter to the reciprocal of a frequency response
    for which associated uncertainties are given for its np.real and np.imaginary part.
    Uncertainties are propagated unp.sing np.linalg.svd and linear matrix propagation.
    
    :param H: frequency response values
    :param UH: matrix of uncertainties associated with the np.real and np.imaginary part of H
    :param N: FIR filter order
    :param tau: delay of filter
    :param f: frequencies
    :param Fs: sampling frequency of digital filter
    
    optional input parameters
    
    :param wt: vector of weights (length 2K) or string 'unc' for unp.sing uncertainties, default=None    
    
    :type H: ndarray, shape (K,)
    :type UH: ndarray, shape (2K,2K)
    :type N: int
    :type tau: int
    :type f: ndarray, shape (K,)
    :type Fs: float
    
    :returns b: filter coefficients of shape (N,)
    :returns Ub: matrix of uncertainties associated with b. shape (N,N)
    
    """
        
    if verbose:
        print "\nLeast-squares fit of an order %d digital FIR filter to the" % N
        print "reciprocal of a frequency response given by %d values" % len(H)
        print "and propagation of associated uncertainties."

  
  # Step 1: Propagation of uncertainties to reciprocal of frequency response
    runs = 10000
    HRI  = np.random.multivariate_normal(np.hstack((np.real(H),np.imag(H))),UH,runs)
    omtau = 2*np.pi*f/Fs*tau
    Nf = len(f)
    absHMC= HRI[:,:Nf]**2 + HRI[:,Nf:]**2
  # Monte Carlo vectorized
    HiMC = np.hstack(((HRI[:,:Nf]*np.tile(np.cos(omtau),(runs,1)) + HRI[:,Nf:]*np.tile(np.sin(omtau),(runs,1)))/absHMC, \
                     (HRI[:,Nf:]*np.tile(np.cos(omtau),(runs,1)) - HRI[:,:Nf]*np.tile(np.sin(omtau),(runs,1)))/absHMC ) )
    UiH = np.cov(HiMC,rowvar=0)

#    if inp.sinstance(wt,str) and wt=='unc':
#        wt = np.sqrt(np.np.diag(UiH)**(-1))
        
#    if wt.shape != np.diag(UiH).shape:
#        raise ValueError("User provided weighting has wrong dimensions")        
    wt = np.ones(2*Nf)        
        
    E = np.exp(-1j*2*np.pi*np.dot(f[:,np.newaxis]/Fs,np.arange(N+1)[:,np.newaxis].T))    
    X = np.vstack((np.real(E),np.imag(E)))
    X = np.dot(np.diag(wt),X)
    Hm= H*np.exp(1j*2*np.pi*f/Fs*tau)
    Hri = np.hstack((np.real(1.0/Hm),np.imag(1.0/Hm)))
    
    u,s,v = np.linalg.svd(X,full_matrices=False)
    if isinstance(trunc_svd_tol,float):
        s[s< trunc_svd_tol] = 0.0
    StSInv = np.zeros_like(s)
    StSInv[s>0] = s[s>0]**(-2)
            
    M = np.dot(
            np.dot(
                    np.dot(v.T,np.diag(StSInv)),
                    np.diag(s)),
            u.T  )
    
    bFIR = np.dot(M,Hri[:,np.newaxis])
    UbFIR= np.dot(np.dot(M,UiH),M.T)
    
    bFIR = bFIR.reshape((N+1,))
    
    if returnHi:
        return bFIR,UbFIR,UiH
    else:
        return bFIR, UbFIR
    
    


    
def LSFIR_uncMC(H,UH,N,tau,f,Fs,wt=None,verbose=True):
    """
    Least-squares fit of a digital FIR filter to the reciprocal of a frequency response
    for which associated uncertainties are given for its np.real and np.imaginary part.
    Uncertainties are propagated unp.sing Monte Carlo method.
    
    :param H: frequency response values
    :param UH: matrix of uncertainties associated with the np.real and np.imaginary part of H
    :param N: FIR filter order
    :param tau: delay of filter
    :param f: frequencies
    :param Fs: sampling frequency of digital filter
    :param wt: (optional) vector of weights 
    
    :type H: ndarray, shape (K,)
    :type UH: ndarray, shape (2K,2K)
    :type N: int
    :type tau: int
    :type f: ndarray, shape (K,)
    :type Fs: float
    
    :returns b: filter coefficients of shape (N,)
    :returns Ub: matrix of uncertainties associated with b. shape (N,N)
    
    """
        
    import numpy as np
    
    if verbose:
        print "\nLeast-squares fit of an order %d digital FIR filter to the" % N
        print "reciprocal of a frequency response given by %d values" % len(H)
        print "and propagation of associated uncertainties."

    
  
  # Step 1: Propagation of uncertainties to reciprocal of frequency response
    runs = 10000
    HRI  = np.random.np.random.multivariate_normal(np.np.hstack((np.np.real(H),np.np.imag(H))),UH,runs)
        
    E = np.np.exp(-1j*2*np.np.pi*np.np.dot(f[:,np.np.newaxis]/Fs,np.np.arange(N+1)[:,np.np.newaxis].T))
    X = np.np.vstack((np.np.real(E),np.np.imag(E)))
    
    Nf = len(f)
    bF= np.zeros((N+1,runs))
    resn =np.zeros((runs,))
    for k in range(runs):
        Hk = HRI[k,:Nf] + 1j*HRI[k,Nf:]
        Hkt= Hk*np.np.exp(1j*2*np.np.pi*f/Fs*tau)
        iRI= np.np.hstack([np.np.real(1.0/Hkt),np.np.imag(1.0/Hkt)])
        bF[:,k],res = np.linalg.lstsq(X,iRI)[:2]
        resn[k]= np.linalg.norm(res)


    bFIR = np.mean(bF,axis=1)
    UbFIR= np.np.cov(bF,rowvar=1)    
   
    return bFIR, UbFIR    
    
    
def LSIIR_unc(H,UH,Nb,Na,f,Fs,tau=0):
    """
    Least-squares fit of a digital IIR filter to the reciprocal of a given set 
    of frequency response values with given associated uncertainty.
    
    :param H: frequency response values.
    :param UH: matrix of uncertainties associated with np.real and np.imaginary part of H
    :param Nb: order of IIR numerator polynomial.
    :param Na: order of IIR denominator polynomial.
    :param f: frequencies.
    :param Fs: sampling frequency for digital IIR filter.
    :param tau: initial estimate of time delay for filter stabilization.

    :type H: ndarray of shape (N,)
    :type UH: ndarray of shape (2N,2N)
    :type Nb: int
    :type Na: int
    :type f: ndarray
    :type Fs: float
    :type tau: int 
    
    :returns b,a: ndarray IIR filter coefficients
    :returns tau: time delay (in samples)
    :returns Uba: uncertainties associated with :math:`(a_1,...,a_{N_a},b_0,...,b_{N_b})`
     
    .. seealso:: :mod:`ADM.examples.deconvolution`
    """
    import numpy as np
    
    runs = 1000

    print "\nLeast-squares fit of an order %d digital IIR filter to the" % max(Nb,Na)
    print "reciprocal of a frequency response given by %d values.\n" % len(H)
    print "Uncerainties of the filter coefficients are evaluated unp.sing\n"\
          "the GUM S2 Monte Carlo method with %d runs.\n" % runs
  
    
    HRI = np.random.multivariate_normal(np.hstack((np.real(H),np.imag(H))),UH,runs)

    HH  = HRI[:,:len(f)] + 1j*HRI[:,len(f):]  
  
    AB = np.zeros((runs,Nb+Na+1))
    Tau= np.zeros((runs,))
    for k in range(runs):        
        bi,ai,Tau[k] = LSIIR(HH[k,:],Nb,Na,f,Fs,tau,verbose=False)
        AB[k,:] = np.hstack((ai[1:],bi))
        
    bi = np.mean(AB[:,Na:],axis=0)
    ai = np.hstack((np.array([1.0]),np.mean(AB[:,:Na],axis=0)))
    Uab= np.cov(AB,rowvar=0)
    tau = np.mean(Tau)
    return bi,ai, tau, Uab
    

def FreqResp2RealImag(Abs,Phase,Unc,MCruns=1e4):
    """
    Calculation of np.real and np.imaginary part from amplitude and phase with associated
    uncertainties.
    
    :param Abs: ndarray of shape N - amplitude values
    :param Phase: ndarray of shape N - phase values in rad
    :param Unc: ndarray of shape 2Nx2N or 2N - uncertainties 
    
    :returns Re,Im: ndarrays of shape N - np.real and np.imaginary part (best estimate)
    :returns URI: ndarray of shape 2Nx2N - uncertainties assoc. with Re and Im
    """
    
    if len(Abs) != len(Phase) or 2*len(Abs) != len(Unc):
        raise ValueError('\nLength of inputs are inconsistent.')
    
    if len(Unc.shape)==1:
        Unc = np.diag(Unc)
        
    Nf = len(Abs)
    
    AbsPhas = np.random.multivariate_normal(np.hstack((Abs,Phase)),Unc,int(MCruns))
    
    H = AbsPhas[:,:Nf]*np.exp(1j*AbsPhas[:,Nf:])
    RI= np.hstack((np.real(H),np.imag(H)))
    
    Re = np.mean(RI[:,:Nf])
    Im = np.mean(RI[:,Nf:])
    URI= np.cov(RI,rowvar=False)
    
    return Re,Im, URI

    
    
    
    
    