# -*- coding: utf-8 -*-
"""
The propagation of uncertainties via the FIR and IIR formulae alone does not
enable the derivation of credible intervals, because the underlying
distribution remains unknown. The GUM-S2 Monte Carlo method provides a
reference method for the calculation of uncertainties for such cases.

This module implements Monte Carlo methods for the propagation of
uncertainties for digital filtering.

This module contains the following functions:

* *MC*: Standard Monte Carlo method for application of digital filter
* *SMC*: Sequential Monte Carlo method with reduced computer memory requirements
* *UMC*: Update Monte Carlo method with reduced computer memory requirements

"""

import sys
import types
import math
import multiprocessing
import functools

import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.signal import lfilter
from scipy.interpolate import interp1d

from ..misc.filterstuff import isstable
from ..misc.tools import progressBar

__all__ = ["MC", "SMC", "UMC"]


class Normal_ZeroCorr:
    """     Multivariate normal distribution with zero correlation"""
    def __init__(self, loc=np.zeros(1), scale=np.zeros(1)):
        """
        Parameters
        ----------
            loc: np.ndarray, optional
                mean values, default is zero
            scale: np.ndarray, optional
                standard deviations for the elements in loc, default is zero
        """

        if isinstance(loc, np.ndarray) or isinstance(scale, np.ndarray):

            # convert loc to array if necessary
            if not isinstance(loc, np.ndarray):
                self.loc = loc * np.ones(1)
            else:
                self.loc = loc

            # convert scale to arraym if necessary
            if not isinstance(scale, np.ndarray):
                self.scale = scale * np.ones(1)
            else:
                self.scale = scale

            # if one of both (loc/scale) has length one, make it bigger to fit
            # size of the other
            if self.loc.size != self.scale.size:
                Nmax = max(self.loc.size, self.scale.size)

                if self.loc.size == 1 and self.scale.size != 1:
                    self.loc = self.loc * np.ones(Nmax)

                elif self.scale.size == 1 and self.loc.size != 1:
                    self.scale = self.scale * np.ones(Nmax)

                else:
                    raise ValueError(
                        "loc and scale do not have the same dimensions. (And "
                        "none of them has dim == 1)")

        else:
            raise TypeError("At least one of loc or scale must be of type "
                            "numpy.ndarray.")

    def rvs(self, size=1):
        # This function mimics the behavior of the scipy stats package
        return np.tile(self.loc, (size, 1)) + \
               np.random.randn(size, len(self.loc)) * \
               np.tile(self.scale, (size, 1))


def MC(
        x, Ux, b, a, Uab, runs=1000, blow=None, alow=None,
        return_samples=False, shift=0, verbose=True
):
    r"""Standard Monte Carlo method

    Monte Carlo based propagation of uncertainties for a digital filter (b,a)
    with uncertainty matrix :math:`U_{\theta}` for
    :math:`\theta=(a_1,\ldots,a_{N_a},b_0,\ldots,b_{N_b})^T`

    Parameters
    ----------
        x: np.ndarray
            filter input signal
        Ux: float or np.ndarray
            standard deviation of signal noise (float), point-wise standard
            uncertainties or covariance matrix associated with x
        b: np.ndarray
            filter numerator coefficients
        a: np.ndarray
            filter denominator coefficients
        Uab: np.ndarray
            uncertainty matrix :math:`U_\theta`
        runs: int,optional
            number of Monte Carlo runs
        return_samples: bool, optional
            whether samples or mean and std are returned

    If ``return_samples`` is ``False``, the method returns:

    Returns
    -------
        y: np.ndarray
            filter output signal
        Uy: np.ndarray
            uncertainty associated with

    Otherwise the method returns:

    Returns
    -------
        Y: np.ndarray
            array of Monte Carlo results

    References
    ----------
        * Eichstädt, Link, Harris and Elster [Eichst2012]_
    """

    Na = len(a)
    runs = int(runs)

    Y = np.zeros((runs, len(x)))   # set up matrix of MC results
    theta = np.hstack((a[1:], b))  # create the parameter vector from the filter coefficients
    Theta = np.random.multivariate_normal(theta, Uab, runs)  # Theta is small and thus we

    # can draw the full matrix now.
    if isinstance(Ux, np.ndarray):
        if len(Ux.shape) == 1:
            dist = Normal_ZeroCorr(loc=x, scale=Ux)  # non-iid noise w/o correlation
        else:
            dist = stats.multivariate_normal(x, Ux)  # colored noise
    elif isinstance(Ux, float):
        dist = Normal_ZeroCorr(loc=x, scale=Ux)      # iid noise
    else:
        raise NotImplementedError("The supplied type of uncertainty is not implemented")

    unst_count = 0  # Count how often in the MC runs the IIR filter is unstable.
    st_inds = list()
    if verbose:
        sys.stdout.write('MC progress: ')
    for k in range(runs):
        xn = dist.rvs()  # draw filter input signal
        if not blow is None:
            if alow is None:
                alow = 1.0  # FIR low-pass filter
            xn = lfilter(blow, alow, xn)  # low-pass filtered input signal
        bb = Theta[k, Na - 1:]
        aa = np.hstack((1.0, Theta[k, :Na - 1]))
        if isstable(bb, aa):
            Y[k, :] = lfilter(bb, aa, xn)
            st_inds.append(k)
        else:
            unst_count += 1  # don't apply the IIR filter if it's unstable
        if np.mod(k, 0.1 * runs) == 0 and verbose:
            sys.stdout.write(' %d%%' % (np.round(100.0 * k / runs)))
    if verbose:
        sys.stdout.write(" 100%\n")

    if unst_count > 0:
        print("In %d Monte Carlo %d filters have been unstable" % (
            runs, unst_count))
        print(
            "These results will not be considered for calculation of mean and "
            "std")
        print(
            "However, if return_samples is 'True' then ALL samples are "
            "returned.")

    Y = np.roll(Y, int(shift), axis=1)  # correct for the (known) sample delay

    if return_samples:
        return Y
    else:
        y = np.mean(Y[st_inds, :], axis=0)
        uy = np.cov(Y[st_inds, :], rowvar=False)
        return y, uy


def SMC(
        x, noise_std, b, a, Uab=None, runs=1000, Perc=None, blow=None,
        alow=None, shift=0, return_samples=False, phi=None, theta=None,
        Delta=0.0
):
    r"""Sequential Monte Carlo method

    Sequential Monte Carlo propagation for a digital filter (b,a) with
    uncertainty matrix :math:`U_{\theta}` for
    :math:`\theta=(a_1,\ldots,a_{N_a},b_0,\ldots,b_{N_b})^T`

    Parameters
    ----------
        x: np.ndarray
            filter input signal
        noise_std: float
            standard deviation of signal noise
        b: np.ndarray
            filter numerator coefficients
        a: np.ndarray
            filter denominator coefficients
        Uab: np.ndarray
            uncertainty matrix :math:`U_\theta`
        runs: int, optional
            number of Monte Carlo runs
        Perc: list, optional
            list of percentiles for quantile calculation
        blow: np.ndarray
            optional low-pass filter numerator coefficients
        alow: np.ndarray
            optional low-pass filter denominator coefficients
        shift: int
            integer for time delay of output signals
        return_samples: bool, otpional
            whether to return y and Uy or the matrix Y of MC results
        phi, theta: np.ndarray, optional
            parameters for AR(MA) noise model
            :math:`\epsilon(n)  = \sum_k \phi_k\epsilon(n-k) + \sum_k
            \theta_k w(n-k) + w(n)` with :math:`w(n)\sim N(0,noise_std^2)`
        Delta: float,optional
             upper bound on systematic error of the filter

    If ``return_samples`` is ``False``, the method returns:

    Returns
    -------
        y: np.ndarray
            filter output signal (Monte Carlo mean)
        Uy: np.ndarray
            uncertainties associated with y (Monte Carlo point-wise std)
        Quant: np.ndarray
            quantiles corresponding to percentiles ``Perc`` (if not ``None``)

    Otherwise the method returns:

    Returns
    -------
        Y: np.ndarray
            array of all Monte Carlo results

    References
    ----------
        * Eichstädt, Link, Harris, Elster [Eichst2012]_
    """

    runs = int(runs)

    if isinstance(a, np.ndarray):  # filter order denominator
        Na = len(a) - 1
    else:
        Na = 0
    if isinstance(b, np.ndarray):  # filter order numerator
        Nb = len(b) - 1
    else:
        Nb = 0

    # Initialize noise matrix corresponding to ARMA noise model.
    if isinstance(theta, np.ndarray) or isinstance(theta, float):
        MA = True
        if isinstance(theta, float):
            W = np.zeros((runs, 1))
        else:
            W = np.zeros((runs, len(theta)))
    else:
        MA = False  # no moving average part in noise process

    # Initialize for autoregressive part of noise process.
    if isinstance(phi, np.ndarray) or isinstance(phi, float):
        AR = True
        if isinstance(phi, float):
            E = np.zeros((runs, 1))
        else:
            E = np.zeros((runs, len(phi)))
    else:
        AR = False  # No autoregressive part in noise process.

    # Initialize matrix of low-pass filtered input signal.
    if isinstance(blow, np.ndarray):
        X = np.zeros((runs, len(blow)))
    else:
        X = np.zeros(runs)

    if isinstance(alow, np.ndarray):
        Xl = np.zeros((runs, len(alow) - 1))
    else:
        Xl = np.zeros((runs, 1))

    if Na == 0:  # only FIR filter
        coefs = b
    else:
        coefs = np.hstack((a[1:], b))

    if isinstance(Uab, np.ndarray):  # Monte Carlo draw for filter coefficients
        Coefs = np.random.multivariate_normal(coefs, Uab, runs)
    else:
        Coefs = np.tile(coefs, (runs, 1))

    b0 = Coefs[:, Na]

    if Na > 0:  # filter is IIR
        A = Coefs[:, :Na]
        if Nb > Na:
            A = np.hstack((A, np.zeros((runs, Nb - Na))))
    else:  # filter is FIR -> zero state equations
        A = np.zeros((runs, Nb))

    # Fixed part of state-space model.
    c = Coefs[:, Na + 1:] - np.multiply(np.tile(b0[:, np.newaxis], (1, Nb)), A)
    States = np.zeros(np.shape(A))  # initialise matrix of states

    calcP = False  # by default no percentiles requested
    if Perc is not None:  # percentiles requested
        calcP = True
        P = np.zeros((len(Perc), len(x)))

    # Initialize outputs.
    y = np.zeros_like(x)
    # Initialize vector of uncorrelated point-wise uncertainties.
    Uy = np.zeros_like(x)

    # Start of the actual MC part.
    print("Sequential Monte Carlo progress", end="")

    for index in np.ndenumerate(x):

        w = np.random.randn(runs) * noise_std  # noise process draw
        if AR and MA:
            E = np.hstack((E.dot(phi) + W.dot(theta) + w, E[:-1]))
            W = np.hstack((w, W[:-1]))
        elif AR:
            E = np.hstack((E.dot(phi) + w, E[:-1]))
        elif MA:
            E = W.dot(theta) + w
            W = np.hstack((w, W[:-1]))
        else:
            w = np.random.randn(runs, 1) * noise_std
            E = w

        if isinstance(alow, np.ndarray):  # apply low-pass filter
            X = np.hstack((x[index] + E, X[:, :-1]))
            Xl = np.hstack(
                (X.dot(blow.T) - Xl[:, :len(alow)].dot(alow[1:]), Xl[:, :-1]))
        elif isinstance(blow, np.ndarray):
            X = np.hstack((x[index] + E, X[:, :-1]))
            Xl = X.dot(blow)
        else:
            Xl = x[index] + E

        # Prepare for easier calculations.
        if len(Xl.shape) == 1:
            Xl = Xl[:, np.newaxis]

        # State-space system output.
        Y = np.sum(np.multiply(c, States), axis=1) + \
            np.multiply(b0, Xl[:, 0]) + \
            (np.random.rand(runs) * 2 * Delta - Delta)
        # Calculate state updates.
        Z = -np.sum(np.multiply(A, States), axis=1) + Xl[:, 0]
        # Store state updates and remove old ones.
        States = np.hstack((Z[:, np.newaxis], States[:, :-1]))

        y[index] = np.mean(Y)  # point-wise best estimate
        Uy[index] = np.std(Y)  # point-wise standard uncertainties
        if calcP:
            P[:, index] = sp.stats.mstats.mquantiles(np.asarray(Y), prob=Perc)

        if np.mod(index, np.round(0.1 * len(x))) == 0:
            print(' %d%%' % (np.round(100.0 * index / len(x))), end="")

    print(" 100%")

    # Correct for (known) delay.
    y = np.roll(y, int(shift))
    Uy = np.roll(Uy, int(shift))

    if calcP:
        P = np.roll(P, int(shift), axis=1)
        return y, Uy, P
    else:
        return y, Uy


def UMC(x, b, a, Uab, runs = 1000, blocksize = 8, blow = [1], alow = [1],
        phi = [0], theta = [0], sigma = 1, Delta = 0.0, runs_init = 100, nbins=[1000], verboseReturn = False):
    """
    Batch Monte Carlo for filtering using update formulae for mean, variance and (approximated) histogram

    Parameters
    ----------
        x: np.ndarray
            filter input signal
        b: np.ndarray
            filter numerator coefficients
        a: np.ndarray
            filter denominator coefficients
        Uab: np.ndarray
            uncertainty matrix :math:`U_\theta`
        runs: int, optional
            number of Monte Carlo runs
        blocksize: int, optional
            how many samples should be evaluated for at a time
        blow: list or np.ndarray, optional
            filter coefficients of optional low pass filter
        alow: list or np.ndarray, optional
            filter coefficients of optional low pass filter
        phi: np.ndarray, optional
            see ARMA noise model
        theta: np.ndarray, optional
            see ARMA noise model
        sigma: float, optional
            see ARMA noise model
        Delta: float, optional
            upper bound of systematic correction due to regularisation (assume uniform distribution)
        runs_init: int, optional
            how many samples to evaluate to form initial guess about limits
        nbins: list of int, optional
            number of bins for histogram
        verboseReturn: bool, optional
            see return-value of documentation

    ARMA noise model
    ----------------
    * eps(n) = sum phi_k*eps(n-k) + sum theta_k*w(n-k) + w(n)
    * with w(n) ~ N(0,sigma^2)

    Returns (if not verboseReturn, default)
    -------
        y: np.ndarray
            filter output signal
        Uy: np.ndarray
            uncertainty associated with

    Returns (if verboseReturn)
    -------
        y: np.ndarray
            filter output signal
        Uy: np.ndarray
            uncertainty associated with
        p025: np.ndarray
            lower 95% credible interval
        p975: np.ndarray
            upper 95% credible interval
        happr: dict
            dictionary keys: given nbin
            dictionary values: bin-edges val["ed"], bin-counts val["f"]

    References
    ----------
        * Eichstädt, Link, Harris, Elster [Eichst2012]_
        * ported to python in 2019-08 from matlab-version of Sascha Eichstaedt (PTB) from 2011-10-12
        * copyright on updating formulae parts is by Peter Harris (NPL)
    """

    # init parallel computation
    nPool = min(multiprocessing.cpu_count(), blocksize)
    pool = multiprocessing.Pool(nPool)

    # ------------ preparations for update formulae ------------

    # set low pass filter
    if alow[0] != 1:
        blow = blow / alow[0]
        alow = alow / alow[0]

    ab = np.hstack((a[1:], b))    # create the parameter vector from the filter coefficients (should be named theta, but this name is already used)

    # init Y
    Y = np.zeros((runs_init, x.size))  # set up matrix of MC results
 
    # define functions to draw samples from (with predefined PDF associated)
    p_ab = lambda size: np.random.multivariate_normal(ab, Uab, size)
    
    # init samples to be evaluated
    TH = p_ab(runs_init)

    # evaluate the initial samples
    for k, res in enumerate(pool.imap(functools.partial(UMCevaluate, nbb=b.size, x=x, Delta=Delta, theta=theta, phi=phi, sigma=sigma, blow=blow, alow=alow), TH)):
        Y[k,:] = res
        progressBar(k, runs_init, prefix="UMC initialisation:     ")
    print("\n") # to escape the carriage-return of progressBar

    # bin edges
    ymin = np.min(Y, axis=0)
    ymax = np.max(Y, axis=0)

    happr = {}
    for nbin in nbins:
        happr[nbin] = {}
        happr[nbin]["ed"] = np.linspace(ymin, ymax, num=nbin+1)  # define bin-edges (generates array for all [ymin,ymax] (assume ymin is already an array))
        happr[nbin]["f"] = np.zeros((nbin, len(x)))              # init. bin-counts

    # ----------------- run MC block-wise -----------------------
    
    blocks = math.ceil(runs/blocksize)

    for m in range(blocks):
        curr_block = min(blocksize, runs - m * blocksize)
        Y = np.zeros((curr_block, len(x)))
        TH = p_ab(curr_block)  # NOTE: change from blocksize to curr_block

        # serial loop
        #for k in range(curr_block): # NOTE: changed loop-count from blocksize to curr_block
        #    th = TH[k,:]
        #    Y[k,:] = UMCevaluate(th, b.size, x, Delta, theta, phi, sigma, blow, alow)

        # parallel loop
        for k, res in enumerate(pool.imap_unordered(functools.partial(UMCevaluate, nbb=b.size, x=x, Delta=Delta, theta=theta, phi=phi, sigma=sigma, blow=blow, alow=alow), TH, curr_block)):
            Y[k,:] = res

        if m == 0: # first block
            y  = np.mean(Y, axis=0)
            uy = np.std(Y, axis=0)

            if verboseReturn:
                for k in range(x.size):
                    for h in happr.values():  # NOTE: this loops over a different index than the matlab-script
                        h["f"][:,k] = np.histogram(Y[:,k], bins = h["ed"][:,k])[0]  # numpy histogram returns (bin-counts, bin-edges)

                ymin = np.min(Y)
                ymax = np.max(Y)

        else: # updating y and uy from results of current block
            K  = m * blocksize
            K0 = curr_block

            # diff to current calculated mean
            d = np.sum(Y - y, axis=0) / (K + K0)

            # new mean
            y = y + d

            # new variance
            s2 = ((K-1)*np.square(uy) + K*np.square(d) + np.sum(np.square(Y-y))) / (K+K0-1)
            uy = np.sqrt(s2)

            if verboseReturn:
                # update histogram values
                for k in range(x.size):
                    for h in happr.values():  # NOTE: this loops over a different index than the matlab-script
                        h["f"][:,k] = np.histogram(Y[:,k], bins = h["ed"][:,k])[0]  # numpy histogram returns (bin-counts, bin-edges)

                ymin = np.min(np.append(ymin,Y))
                ymax = np.max(np.append(ymax,Y))

        progressBar(m*blocksize, runs, prefix="UMC running:            ")  # spaces on purpose, to match length of progress-bar below
    print("\n") # to escape the carriage-return of progressBar
    

    # ----------------- post-calculation steps -----------------------

    if verboseReturn:
        # remove the last frequency, which is always zero
        for nbin, h in happr.items():
            h["f"] = h["f"][:-1,:]
            h["ed"][0,:]  = np.min(np.append(ymin, h["ed"][1,:]))
            h["ed"][-1,:] = np.max(np.append(ymax, h["ed"][-2,:]))

        # replace edge limits by ymin and ymax, resp.
        p025 = np.zeros((len(nbins), len(y)))
        p975 = np.zeros((len(nbins), len(y)))

        for k in range(x.size):
            for m, h in enumerate(happr.values()):
                e = h["ed"][:-1,k]  # last bin-edge is not required
                f = h["f"][:,k]
                G = np.append(0, np.cumsum(f)/np.sum(f))

                # quick fix to ensure strictly increasing G
                iz = np.argwhere(np.diff(G) == 0)
                if iz.size != 0:
                    for l in iz:   # NOTE: is there a wrong index in the matlab-script?
                        G[l+1]  = G[l] + 10*np.spacing(1.0)   # numpy.spacing gives the machine-epsilon

                pcov = np.linspace(G[0], G[-1]-0.95, 100)

                ylow = interp1d(G,e)(pcov)
                yhgh = interp1d(G,e)(pcov+0.95)

                lcov = yhgh - ylow
                imin = np.argmin(lcov)

                p025[m,k] = ylow[imin]
                p975[m,k] = yhgh[imin]
            
            progressBar(k, x.size,prefix="UMC credible intervals: ")
        print("\n") # to escape the carriage-return of progressBar
        
        return y,uy,p025,p975,happr

    else:
        return y, uy


def UMCevaluate(th, nbb, x, Delta, theta, phi, sigma, blow, alow):

    naa = len(th) - nbb + 1        # theta contains all but the first entry of aa
    aa = np.append(1, th[:naa-1])  # insert coeff 1 at position 0 to restore aa
    bb = th[naa-1:]                # restore bb

    e = ARMA(x.size, phi = phi, theta = theta, std = sigma)

    xlow = lfilter(blow, alow, x + e)
    d = Delta * (2 * np.random.random_sample(size=x.size) - 1 )   # uniform distribution [-Delta, Delta]
    res = lfilter(bb, aa, xlow) + d
    
    return res


# move this function to ..misc.noise.ARMA
def ARMA(length, phi = [0], theta = [0], std = 1):
    """
    Generate time-series of a predefined ARMA-process based on this equation:
    :math:`\sum_{j=1}^{\min(p,n-1)} \phi_j \epsilon[n-j] + \sum_{j=1}^{\min(q,n-1)} \\nu_j w[n-j]`
    Equation and algorithm taken from [Eichst2012]_

    Parameters
    ----------
    length: int
        how long the drawn sample will be
    phi: list or numpy.ndarray
        AR-coefficients
    theta: list or numpy.ndarray
        MA-coefficients
    std: float
        std of the gaussian white noise that is feeded into the ARMA-model
        
    References
    ----------
        * Eichstädt, Link, Harris and Elster [Eichst2012]_
    """

    # convert to numpy.ndarray
    if isinstance(phi, list): phi = np.array(phi)
    if isinstance(theta, list): theta = np.array(theta)

    # initialize e, w
    w = np.random.normal(loc = 0, scale = std, size = length)
    e = np.zeros_like(w)

    # define shortcuts
    p = len(phi)
    q = len(theta)
    
    # iterate series over time
    for n, wn in enumerate(w):
        min_pn = min(p, n)
        min_qn = min(q, n)
        e[n] = np.sum(phi[:min_pn].dot(e[n-min_pn:n])) + np.sum(theta[:min_qn].dot(w[n-min_qn:n])) + wn

    return e