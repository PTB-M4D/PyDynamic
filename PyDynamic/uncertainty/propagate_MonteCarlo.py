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

import functools
import math
import multiprocessing
import sys

import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.signal import lfilter

from ..misc.filterstuff import isstable
from ..misc.tools import progress_bar

__all__ = ["MC", "SMC", "UMC", "UMC_generic"]


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


def UMC(x, b, a, Uab, runs = 1000, blocksize = 8, blow = 1.0, alow = 1.0,
        phi = 0.0, theta = 0.0, sigma = 1, Delta = 0.0, runs_init = 100, nbins=1000, verbose_return = False):
    """
    Batch Monte Carlo for filtering using update formulae for mean, variance and (approximated) histogram.
    This is a wrapper for the UMC_generic function, specialised on filters

    Parameters
    ----------
        x: np.ndarray, shape (nx, )
            filter input signal
        b: np.ndarray, shape (nbb, )
            filter numerator coefficients
        a: np.ndarray, shape (naa, )
            filter denominator coefficients, normalization (a[0] == 1.0) is assumed
        Uab: np.ndarray, shape (naa + nbb - 1, )
            uncertainty matrix :math:`U_\\theta`
        runs: int, optional
            number of Monte Carlo runs
        blocksize: int, optional
            how many samples should be evaluated for at a time
        blow: float or np.ndarray, optional
            filter coefficients of optional low pass filter
        alow: float or np.ndarray, optional
            filter coefficients of optional low pass filter
        phi: np.ndarray, optional,
            see misc.noise.ARMA noise model
        theta: np.ndarray, optional
            see misc.noise.ARMA noise model
        sigma: float, optional
            see misc.noise.ARMA noise model
        Delta: float, optional
            upper bound of systematic correction due to regularisation (assume uniform distribution)
        runs_init: int, optional
            how many samples to evaluate to form initial guess about limits
        nbins: int, list of int, optional
            number of bins for histogram
        verbose_return: bool, optional
            see return-value of documentation

    By default, phi, theta, sigma are choosen such, that N(0,1)-noise is added to the input signal.

    Returns (if not verbose_return, default)
    -------
        y: np.ndarray
            filter output signal
        Uy: np.ndarray
            uncertainty associated with

    Returns (if verbose_return)
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
            dictionary values: bin-edges val["bin-edges"], bin-counts val["bin-counts"]

    References
    ----------
        * Eichstädt, Link, Harris, Elster [Eichst2012]_
        * ported to python in 2019-08 from matlab-version of Sascha Eichstaedt (PTB) from 2011-10-12
        * copyright on updating formulae parts is by Peter Harris (NPL)
    """

    # input adjustments and type conversions
    if isinstance(alow, float):
        alow = np.array([alow])
    if isinstance(blow, float):
        blow = np.array([blow])
    if isinstance(nbins, int):
        nbins = [nbins]

    if alow[0] != 1:
        blow = blow / alow[0]
        alow = alow / alow[0]


    # define generic functions to hand over to UMC_generic

    # variate the coefficients of filter as main simulation influence
    ab = np.hstack((a[1:], b))    # create the parameter vector from the filter coefficients (should be named theta, but this name is already used)
    drawSamples = lambda size: np.random.multivariate_normal(ab, Uab, size)

    # how to evaluate functions
    params = {"nbb": b.size,
              "x": x,
              "sigma": sigma,
              "blow": blow,
              "alow": alow,
              "Delta": Delta,
              "phi": phi,
              "theta": theta}
    evaluate = functools.partial(_UMCevaluate, **params)

    # run UMC
    y, Uy, happr = UMC_generic(drawSamples, evaluate, runs=runs, blocksize=blocksize, runs_init=runs_init)

    # further post-calculation steps
    if verbose_return:

        p025 = np.zeros((len(nbins), len(y)))
        p975 = np.zeros((len(nbins), len(y)))

        # approximate 2.5% and 97.5% percentiles
        for k in range(x.size):
            for m, h in enumerate(happr.values()):
                e = h["bin-edges"][:,k]                 # take all bin-edges
                f = np.append(0, h["bin-counts"][:,k])  # bin count for before first bin is 0
                G = np.cumsum(f)/np.sum(f)

                ## interpolate the cumulated relative bin-count G(e) for the requested credibility interval
                cred = 0.95
                interp_e = interp1d(G,e)

                # save credibility intervals
                p025[m,k] = interp_e((1+cred)/2)
                p975[m,k] = interp_e((1-cred)/2)

        return y, Uy, p025, p975, happr

    else:
        return y, Uy


def _UMCevaluate(th, nbb, x, Delta, phi, theta, sigma, blow, alow):
    """
    Calculate system-response of an IIR-filter to some input signal x.

    Parameters
    ----------
    th: numpy.ndarray, shape (naa + nbb -1, )
        coefficients of an IIR filter, :math:`\\theta = [aa[1:], b]`
    nbb: int
        size of bb within th
    x: numpy.ndarray, shape (nx, )
        input signal
    Delta: float
        add noise to output drawn from uniform-distribution U([-Delta, Delta])
    phi: np.ndarray
        see misc.noise.ARMA noise model
    theta: np.ndarray
        see misc.noise.ARMA noise model
    sigma: float
        see misc.noise.ARMA noise model
    blow: float or np.ndarray
        filter coefficients of low pass filter applied to sum of input-signal and ARMA-noise
    alow: float or np.ndarray
        filter coefficients of low pass filter applied to sum of input-signal and ARMA-noise


    x -----------------+--->[LOWPASS]--->[IIR-FILTER]----+---> y
                       |                                 |
    ARMA(phi,theta) ---'            U([-Delta,Delta]) ---'
    """

    naa = len(th) - nbb + 1        # theta contains all but the first entry of aa
    aa = np.append(1, th[:naa-1])  # insert coeff 1 at position 0 to restore aa
    bb = th[naa-1:]                # restore bb

    e = ARMA(x.size, phi = phi, theta = theta, std = sigma)

    xlow = lfilter(blow, alow, x + e)
    d = Delta * (2 * np.random.random_sample(size=x.size) - 1 )   # uniform distribution [-Delta, Delta]

    return lfilter(bb, aa, xlow) + d


# move this function to ..misc.noise.ARMA
def ARMA(length, phi = 0.0, theta = 0.0, std = 1.0):
    """
    Generate time-series of a predefined ARMA-process based on this equation:
    :math:`\sum_{j=1}^{\min(p,n-1)} \phi_j \epsilon[n-j] + \sum_{j=1}^{\min(q,n-1)} \\theta_j w[n-j]`
    where w is white gaussian noise. Equation and algorithm taken from [Eichst2012]_ .

    Parameters
    ----------
    length: int
        how long the drawn sample will be
    phi: float, list or numpy.ndarray, shape (p, )
        AR-coefficients
    theta: float, list or numpy.ndarray,
        MA-coefficients
    std: float
        std of the gaussian white noise that is feeded into the ARMA-model

    References
    ----------
        * Eichstädt, Link, Harris and Elster [Eichst2012]_
    """

    # convert to numpy.ndarray
    if isinstance(phi, float):
        phi = np.array([phi])
    elif isinstance(phi, list):
        phi = np.array(phi)

    if isinstance(theta, float):
        theta = np.array([theta])
    elif isinstance(theta, list):
        theta = np.array(theta)

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


def UMC_generic(drawSamples, evaluate, runs = 100, blocksize = 8, runs_init = 10, nbins = 100, return_simulations = False, n_cpu = multiprocessing.cpu_count()):
    """
    Generic Batch Monte Carlo using update formulae for mean, variance and (approximated) histogram.
    Assumes that the output of evaluate is a numeric vector.

    Parameters
    ----------
        drawSamples: function(int nDraws)
            function that draws nDraws from a given distribution / population
        evaluate: function(sample)
            function that evaluates a sample and returns the result
        runs: int, optional
            number of Monte Carlo runs
        blocksize: int, optional
            how many samples should be evaluated for at a time
        runs_init: int, optional
            how many samples to evaluate to form initial guess about limits
        return_simulations: bool, optional
            see return-value of documentation
        n_cpu: int
            number of CPUs to use for multiprocessing, defaults to all available CPUs

    Cookbook
    --------
        draw samples from multivariate normal distribution:
            drawSamples = lambda size: np.random.multivariate_normal(x, Ux, size)

        build a function, that only accepts one argument by masking addtional kwargs:
            evaluate = functools.partial(_UMCevaluate, nbb=b.size, x=x, Delta=Delta, phi=phi, theta=theta, sigma=sigma, blow=blow, alow=alow)
            evaluate = functools.partial(bigFunction, **dict_of_kwargs)

    Returns (if not return_simulations)
    -------
        y: np.ndarray
            filter output signal
        Uy: np.ndarray
            uncertainty associated with
        happr: dict
            dictionary of bin-edges and bin-counts

    Returns (if return_simulations)
    -------
        Y: list
            individual results of every individual

    References
    ----------
        * Eichstädt, Link, Harris, Elster [Eichst2012]_
    """

    # type-conversions
    if isinstance(nbins, int):
        nbins = [nbins]

    # init parallel computation
    nPool = min(n_cpu, blocksize)
    pool = multiprocessing.Pool(nPool)

    # ------------ preparations for update formulae ------------

    # set up list of MC results
    Y = [None]*runs_init

    # init samples to be evaluated
    samples = drawSamples(runs_init)

    # evaluate the initial samples
    for k, result in enumerate(pool.imap_unordered(evaluate, samples)):
        progress_bar(k, runs_init, prefix="UMC initialisation:     ")
    print("\n") # to escape the carriage-return of progress_bar

    # convert to array
    Y = np.asarray(Y)

    # get size of in- and output (was so far not explicitly known)
    inputSize = samples.shape[1]
    outputSize = Y.shape[1]

    # prepare histograms
    # bin edges
    ymin = np.min(Y, axis=0)
    ymax = np.max(Y, axis=0)

    happr = {}
    for nbin in nbins:
        happr[nbin] = {}
        happr[nbin]["bin-edges"] = np.linspace(ymin, ymax, num=nbin+1)  # define bin-edges (generates array for all [ymin,ymax] (assume ymin is already an array))
        happr[nbin]["bin-counts"] = np.zeros((nbin, outputSize))        # init. bin-counts

    # ----------------- run MC block-wise -----------------------

    nblocks = math.ceil(runs/blocksize)

    # remember all evaluated simulations, if wanted
    if return_simulations:
        sims = {"params": np.empty((runs, inputSize)), "results": np.empty((runs, outputSize))}

    for m in range(nblocks):
        if m == nblocks:
            curr_block = runs % blocksize
        else:
            curr_block = blocksize

        Y = np.empty((curr_block, outputSize))
        samples = drawSamples(curr_block)

        # parallel loop
        for k, result in enumerate(pool.imap_unordered(evaluate, samples)):
            Y[k,:] = result

        if m == 0: # first block
            y  = np.mean(Y, axis=0)
            Uy = np.matmul((Y-y).T, (Y-y))

            # update histogram values
            for k in range(outputSize):
                for h in happr.values():
                    h["bin-counts"][:,k] = np.histogram(Y[:,k], bins = h["bin-edges"][:,k])[0]  # numpy histogram returns (bin-counts, bin-edges)

            ymin = np.min(Y, axis=0)
            ymax = np.max(Y, axis=0)

        else: # updating y and Uy from results of current block
            K  = m * blocksize
            K0 = curr_block

            # update mean (formula 7 in [Eichst2012])
            y0 = y
            y = y + np.sum(Y - y, axis=0) / (K + K0)

            # update covariance (formula 8 in [Eichst2012])
            Uy = ( (K-1)*Uy + K*np.outer(y-y0, y-y0) + np.matmul((Y-y).T, (Y-y)) ) / (K+K0-1)

            # update histogram values
            for k in range(outputSize):
                for h in happr.values():
                    h["bin-counts"][:,k] += np.histogram(Y[:,k], bins = h["bin-edges"][:,k])[0]  # numpy histogram returns (bin-counts, bin-edges)

            ymin = np.min(np.vstack((ymin,Y)), axis=0)
            ymax = np.max(np.vstack((ymax,Y)), axis=0)

        # save results if wanted
        if return_simulations:
            sims["params"][m*blocksize:m*blocksize+curr_block, :] = samples
            sims["results"][m*blocksize:m*blocksize+curr_block, :] = Y

        progress_bar(m*blocksize, runs, prefix="UMC running:            ")  # spaces on purpose, to match length of progress-bar below
    print("\n") # to escape the carriage-return of progress_bar


    # ----------------- post-calculation steps -----------------------

    # replace edge limits by ymin and ymax, resp.
    for h in happr.values():
        h["bin-edges"][0,:]  = np.min(np.vstack((ymin, h["bin-edges"][0,:])), axis=0)
        h["bin-edges"][-1,:] = np.min(np.vstack((ymax, h["bin-edges"][-1,:])), axis=0)

    if return_simulations:
        return sims
    else:
        return y, Uy, happr
