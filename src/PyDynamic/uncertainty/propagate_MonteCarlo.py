""""Monte Carlo methods for the propagation of uncertainties for digital filtering

The propagation of uncertainties via the FIR and IIR formulae alone does not
enable the derivation of credible intervals, because the underlying
distribution remains unknown. The GUM-S2 Monte Carlo method provides a
reference method for the calculation of uncertainties for such cases.

This module contains the following functions:

* :func:`MC`: Standard Monte Carlo method for application of digital filter
* :func:`SMC`: Sequential Monte Carlo method with reduced computer memory requirements
* :func:`UMC`: Update Monte Carlo method for application of digital filters with
  reduced computer memory requirements
* :func:`UMC_generic`: Update Monte Carlo method with reduced computer memory
  requirements
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
from ..misc.noise import ARMA
from ..misc.tools import progress_bar

__all__ = ["MC", "SMC", "UMC", "UMC_generic"]


class Normal_ZeroCorr:
    """Multivariate normal distribution with zero correlation"""

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
                        "none of them has dim == 1)"
                    )

        else:
            raise TypeError(
                "At least one of loc or scale must be of type " "numpy.ndarray."
            )

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
        dist = Normal_ZeroCorr(loc=x, scale=Ux)  # iid noise
    else:
        raise NotImplementedError("The supplied type of uncertainty is not implemented")

    unst_count = 0  # Count how often in the MC runs the IIR filter is unstable.
    st_inds = list()
    if verbose:
        sys.stdout.write("MC progress: ")
    for k in range(runs):
        xn = dist.rvs()  # draw filter input signal
        if not blow is None:
            if alow is None:
                alow = 1.0  # FIR low-pass filter
            xn = lfilter(blow, alow, xn)  # low-pass filtered input signal
        bb = Theta[k, Na - 1 :]
        aa = np.hstack((1.0, Theta[k, : Na - 1]))
        if isstable(bb, aa):
            Y[k, :] = lfilter(bb, aa, xn)
            st_inds.append(k)
        else:
            unst_count += 1  # don't apply the IIR filter if it's unstable
        if np.mod(k, 0.1 * runs) == 0 and verbose:
            sys.stdout.write(" %d%%" % (np.round(100.0 * k / runs)))
    if verbose:
        sys.stdout.write(" 100%\n")

    if unst_count > 0:
        print("In %d Monte Carlo %d filters have been unstable" % (runs, unst_count))
        print("These results will not be considered for calculation of mean and " "std")
        print("However, if return_samples is 'True' then ALL samples are " "returned.")

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
    c = Coefs[:, Na + 1 :] - np.multiply(np.tile(b0[:, np.newaxis], (1, Nb)), A)
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

    for index, xi in np.ndenumerate(x):

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
            X = np.hstack((xi + E, X[:, :-1]))
            Xl = np.hstack(
                (X.dot(blow.T) - Xl[:, : len(alow)].dot(alow[1:]), Xl[:, :-1])
            )
        elif isinstance(blow, np.ndarray):
            X = np.hstack((xi + E, X[:, :-1]))
            Xl = X.dot(blow)
        else:
            Xl = xi + E

        # Prepare for easier calculations.
        if len(Xl.shape) == 1:
            Xl = Xl[:, np.newaxis]

        # State-space system output.
        Y = (
            np.sum(np.multiply(c, States), axis=1)
            + np.multiply(b0, Xl[:, 0])
            + (np.random.rand(runs) * 2 * Delta - Delta)
        )
        # Calculate state updates.
        Z = -np.sum(np.multiply(A, States), axis=1) + Xl[:, 0]
        # Store state updates and remove old ones.
        States = np.hstack((Z[:, np.newaxis], States[:, :-1]))

        y[index[0]] = np.mean(Y)  # point-wise best estimate
        Uy[index[0]] = np.std(Y)  # point-wise standard uncertainties
        if calcP:
            P[:, index[0]] = sp.stats.mstats.mquantiles(np.asarray(Y), prob=Perc)

        if np.mod(index[0], np.round(0.1 * len(x))) == 0:
            print(" %d%%" % (np.round(100.0 * index[0] / len(x))), end="")

    print(" 100%")

    # Correct for (known) delay.
    y = np.roll(y, int(shift))
    Uy = np.roll(Uy, int(shift))

    if calcP:
        P = np.roll(P, int(shift), axis=1)
        return y, Uy, P
    else:
        return y, Uy


def UMC(
        x, b, a, Uab, runs=1000, blocksize=8, blow=1.0, alow=1.0, phi=0.0,
        theta=0.0, sigma=1, Delta=0.0, runs_init=100, nbins=1000,
        credible_interval=0.95
):
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
        credible_interval: float, optional
            must be in [0,1]
            central credible interval size

    By default, phi, theta, sigma are chosen such, that N(0,1)-noise is added to the input signal.

    Returns
    -------
        y: np.ndarray
            filter output signal
        Uy: np.ndarray
            uncertainty associated with
        y_cred_low: np.ndarray
            lower boundary of credible interval
        y_cred_high: np.ndarray
            upper boundary of credible interval
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
    draw_samples = lambda size: np.random.multivariate_normal(ab, Uab, size)

    # how to evaluate functions
    params = {
        "nbb": b.size,
        "x": x,
        "sigma": sigma,
        "blow": blow,
        "alow": alow,
        "Delta": Delta,
        "phi": phi,
        "theta": theta,
    }
    evaluate = functools.partial(_UMCevaluate, **params)

    # run UMC
    y, Uy, happr, _ = UMC_generic(draw_samples, evaluate, runs=runs, blocksize=blocksize, runs_init=runs_init)

    # further post-calculation steps
    y_cred_low = np.zeros((len(nbins), len(y)))
    y_cred_high = np.zeros((len(nbins), len(y)))

    # approximate lower and upper credible quantiles
    for k in range(x.size):
        for m, h in enumerate(happr.values()):
            e = h["bin-edges"][:,k]                 # take all bin-edges
            f = np.append(0, h["bin-counts"][:,k])  # bin count for before first bin is 0
            G = np.cumsum(f)/np.sum(f)

            ## interpolate the cumulated relative bin-count G(e) for the requested credibility interval
            interp_e = interp1d(G, e)

            # save credibility intervals
            y_cred_low[m, k] = interp_e((1 - credible_interval) / 2)
            y_cred_high[m, k] = interp_e((1 + credible_interval) / 2)

    return y, Uy, y_cred_low, y_cred_high, happr


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

    ```
    x -----------------+--->[LOWPASS]--->[IIR-FILTER]----+---> y
                       |                                 |
    ARMA(phi,theta) ---'            U([-Delta,Delta]) ---'
    ```
    """

    naa = len(th) - nbb + 1  # theta contains all but the first entry of aa
    aa = np.append(1, th[: naa - 1])  # insert coeff 1 at position 0 to restore aa
    bb = th[naa - 1 :]  # restore bb

    e = ARMA(x.size, phi=phi, theta=theta, std=sigma)

    xlow = lfilter(blow, alow, x + e)
    d = Delta * (2 * np.random.random_sample(size=x.size) - 1 )   # uniform distribution [-Delta, Delta]

    return lfilter(bb, aa, xlow) + d


def UMC_generic(draw_samples, evaluate, runs = 100, blocksize = 8, runs_init = 10, nbins = 100,
                return_samples = False, n_cpu = multiprocessing.cpu_count()):
    """
    Generic Batch Monte Carlo using update formulae for mean, variance and (approximated) histogram.
    Assumes that the input and output of evaluate are numeric vectors (but not necessarily of same dimension).
    If the output of evaluate is multi-dimensional, it will be flattened into 1D. 

    Parameters
    ----------
        draw_samples: function(int nDraws)
            function that draws nDraws from a given distribution / population
            needs to return a list of (multi dimensional) numpy.ndarrays
        evaluate: function(sample)
            function that evaluates a sample and returns the result
            needs to return a (multi dimensional) numpy.ndarray
        runs: int, optional
            number of Monte Carlo runs
        blocksize: int, optional
            how many samples should be evaluated for at a time
        runs_init: int, optional
            how many samples to evaluate to form initial guess about limits
        nbins: int, list of int, optional
            number of bins for histogram
        return_samples: bool, optional
            see return-value of documentation
        n_cpu: int, optional
            number of CPUs to use for multiprocessing, defaults to all available CPUs

    Example
    -------
        draw samples from multivariate normal distribution:
        ``draw_samples = lambda size: np.random.multivariate_normal(x, Ux, size)``

        build a function, that only accepts one argument by masking
        additional kwargs:
        ``evaluate = functools.partial(_UMCevaluate, nbb=b.size, x=x, Delta=Delta, phi=phi, theta=theta, sigma=sigma, blow=blow, alow=alow)``
        ``evaluate = functools.partial(bigFunction, **dict_of_kwargs)``

    By default the method 

    Returns
    -------
        y: np.ndarray
            mean of flattened/raveled simulation output
            i.e.: y = np.ravel(evaluate(sample))
        Uy: np.ndarray
            covariance associated with y
        happr: dict
            dictionary of bin-edges and bin-counts
        output_shape: tuple
            shape of the unraveled simulation output
            can be used to reshape y and np.diag(Uy) into original shape

    If ``return_samples`` is ``True``, the method additionally returns all evaluated samples. 
    This should only be done for testing and debugging reasons, as this removes all memory-improvements of the UMC-method. 

    Returns
    -------
        sims: dict
            dict of samples and corresponding results of every evaluated simulation
            samples and results are saved in their original shape

    References
    ----------
        * Eichstädt, Link, Harris, Elster [Eichst2012]_
    """

    # type-conversions
    if isinstance(nbins, int):
        nbins = [nbins]

    # check if parallel computation is required
    # this allows to circumvent a multiprocessing-problem on windows-machines
    # see: https://github.com/PTB-M4D/PyDynamic/issues/84
    if n_cpu == 1:
        map_func = map
    else:
        nPool = min(n_cpu, blocksize)
        pool = multiprocessing.Pool(nPool)
        map_func = pool.imap_unordered

    # ------------ preparations for update formulae ------------

    # set up list of MC results
    Y_init = [None] * runs_init

    # init samples to be evaluated
    samples = draw_samples(runs_init)

    # evaluate the initial samples
    for k, result in enumerate(map_func(evaluate, samples)):
        Y_init[k] = result
        progress_bar(k, runs_init, prefix="UMC initialisation:     ")
    print("\n")  # to escape the carriage-return of progress_bar

    # get size of in- and output (was so far not explicitly known)
    input_shape = samples[0].shape
    output_shape = Y_init[0].shape

    # convert to array
    Y_init = np.asarray(Y_init)

    # prepare histograms
    ymin = np.min(Y_init, axis=0).ravel()
    ymax = np.max(Y_init, axis=0).ravel()

    happr = {}
    for nbin in nbins:
        happr[nbin] = {}
        happr[nbin]["bin-edges"] = np.linspace(ymin, ymax, num=nbin+1)           # define bin-edges (generates array for all [ymin,ymax] (assume ymin is already an array))
        happr[nbin]["bin-counts"] = np.zeros((nbin, np.prod(output_shape)))   # init. bin-counts

    # ----------------- run MC block-wise -----------------------

    nblocks = math.ceil(runs / blocksize)

    # remember all evaluated simulations, if wanted
    if return_samples:
        sims = {"samples": np.empty((runs, *input_shape)), "results": np.empty((runs, *output_shape))}

    for m in range(nblocks):
        if m == nblocks:
            curr_block = runs % blocksize
        else:
            curr_block = blocksize

        Y = np.empty((curr_block, np.prod(output_shape)))
        samples = draw_samples(curr_block)

        # evaluate samples in parallel loop
        for k, result in enumerate(map_func(evaluate, samples)):
            Y[k] = result.ravel()

        if m == 0:  # first block
            y = np.mean(Y, axis=0)
            Uy = np.matmul((Y - y).T, (Y - y))

        else:  # updating y and Uy from results of current block
            K0 = m * blocksize
            K_seq = curr_block

            # update mean (formula 7 in [Eichst2012])
            y0 = y
            y = y0 + np.sum(Y - y0, axis=0) / (K0 + K_seq)

            # update covariance (formula 8 in [Eichst2012])
            Uy = ( (K0-1)*Uy + K0*np.outer(y-y0, y-y0) + np.matmul((Y-y).T, (Y-y)) ) / (K0 + K_seq - 1)

        # update histogram values
        for k in range(np.prod(output_shape)):
            for h in happr.values():
                h["bin-counts"][:,k] += np.histogram(Y[:,k], bins = h["bin-edges"][:,k])[0]  # numpy histogram returns (bin-counts, bin-edges)

        ymin = np.min(np.vstack((ymin, Y)), axis=0)
        ymax = np.max(np.vstack((ymax, Y)), axis=0)

        # save results if wanted
        if return_samples:
            block_start = m * blocksize
            block_end = block_start + curr_block
            sims["samples"][block_start:block_end] = samples
            sims["results"][block_start:block_end] = np.asarray([element.reshape(output_shape) for element in Y])

        progress_bar(m*blocksize, runs, prefix="UMC running:            ")  # spaces on purpose, to match length of progress-bar below
    print("\n") # to escape the carriage-return of progress_bar

    # ----------------- post-calculation steps -----------------------

    # replace edge limits by ymin and ymax, resp.
    for h in happr.values():
        h["bin-edges"][0, :] = np.min(np.vstack((ymin, h["bin-edges"][0, :])), axis=0)
        h["bin-edges"][-1, :] = np.min(np.vstack((ymax, h["bin-edges"][-1, :])), axis=0)

    if return_samples:
        return y, Uy, happr, output_shape, sims
    else:
        return y, Uy, happr, output_shape
