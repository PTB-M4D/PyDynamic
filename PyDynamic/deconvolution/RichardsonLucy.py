# -*- coding: utf-8 -*-
"""
This module contains methods to apply the Richardson-Lucy method for iterative deconvolution as described in

Eichstaedt, Schmaehling, Wuebbeler, Anhalt, Buenger, Krueger & Elster
'Comparison of the Richardson-Lucy method and a classical approach for
spectrometer bandwidth correction', Metrologia vol. 50, 2013
DOI: 10.1088/0026-1394/50/2/107

"""
import sys
import numpy as np

class stopping_rule(object):
    """Base class to set up interface for use of self-defined stopping rules in RichLucy
    """

    crit_measure = None  # measure underlying the stopping rule  (e.g. rms)
    maxiter = None  # maximum number of RL iterations taken
    display = True  # whether to print text to the cmd line
    min_auto_iter = 5  # number of minimal required RL iterations
    name = "(name not defined)" # name of the stopping rule for display in cmd line outputs

    def calc_stoppoint(self, SH):
        """
        Main function to be called for application of stopping rule. Returns
        integer value indicating index of optimal RichLucy result in SH.

        Parameters
        ----------
            SH:  numpy.ndarray of shape (K,N) with K the number of RL iterations and N the number of data points

        Returns
        -------
            ind: integer indicating index of best estimate in SH
        """
        return -1

    def output(self, text):
        "Function to print text to command line if display variable is True"
        if self.display:
            sys.stdout.write(text)

    def plot(self, fig_nr=1):
        "Function to plot result of stopping rule."
        raise NotImplementedError("Plotting not implemented")


class discrepancy(stopping_rule):
    """
    Derivation of a stopping rule from the approximate discrepancy measure from Bardsley et al.
    Published in Eichstädt et al. 2014

    This method is initialized with the following parameters:

    Parameters
    ----------
        M: ndarray of shape (L,) with the measured spectrum values
        btilde: ndarray, bandpass values after discretization and reverse ordering
        dnoise: float, standard deviation of measurement noise
        tau: float, scaling parameter (default 0.9)
    """
    name = "discrepancy principle"

    def __init__(self, M, btilde, dnoise, tau=0.9):
        self.Meas = M
        self.btilde = btilde
        self.dnoise = dnoise
        self.tau = tau
        self.N = len(M)

    def calc_stoppoint(self, SH):
        """Calculation of the optimal stopping point
        Parameters
        ----------
            SH: 2D ndarray of relevant RichLucy results

        Returns
        -------
            int, index of optimal stopping point
        """
        lenSH, lM = np.shape(SH)
        self.maxiter = lenSH - 1

        if lM != self.N:
            raise ValueError('Dimension of estimated spectra and measured spectrum do not fit together.\n')

        self.output('Stopping rule: discrepancy principle.\n')
        self.output('measurement noise std = %g\n' % self.dnoise)
        self.output('scaling parameter = ' + repr(self.tau) + '\n')

        vals = np.zeros(self.maxiter)
        self.output('Calculating relative errors in estimated measurement...')
        for r in range(1, self.maxiter + 1):
            Mh = np.convolve(SH[r, :], self.btilde, 'same')
            vals[r - 1] = (np.linalg.norm((Mh - self.Meas) / self.dnoise) ** 2 - self.tau * lM) ** 2
        self.output('done\n')

        self.crit_measure = vals

        if self.maxiter > self.min_auto_iter:
            if any(np.isnan(vals)) or any(np.isinf(vals)) or np.min(vals) == np.max(vals):
                self.output("Calculation of optimal stopping using the discrepancy principle failed.\n")
                return -1
            indopt = vals.argmin()
            if not (isinstance(indopt, int) or isinstance(indopt, np.int64)):
                indopt = self.maxiter - 1
                self.output("Calculation of optimal stopping using the discrepancy principle failed.\n")
        else:
            self.output('Warning: Maximum number of iterations is too small. Automatic stopping not available.\n')
            indopt = self.maxiter - 1
        self.indopt = indopt
        if indopt > 0:
            self.output('Estimated optimal stopping point is iter: %d \n' % (indopt + 1))
        return indopt

    def plot(self, fig_nr=1):
        from matplotlib.pyplot import figure, clf, semilogx, legend, draw, axvline
        figure(fig_nr)
        clf()
        semilogx(range(1, self.maxiter + 1), np.log10(self.crit_measure), label="discrepancy criterion")
        legend(loc="best")
        axvline(self.indopt, color="k", linestyle="--")
        draw()


def max_iters(noise_std, maxi, absolute_max=20000):
    """ rough mapping of maxiter from noise std and given mapping maxi
example mapping
    maxi = [[-3,1000],
            [-5,2000],
            [-6,5000]]
    """
    nstd = np.log10(noise_std)
    for rel in maxi:
        if nstd >= rel[0]: return rel[1]
    return absolute_max


def calc_log_curv(iters, vals, length=None):
    """ Calculate curvature for logarithmic x-axis
    Parameters
    ----------
        iters: iterations, ndarray of shape N
        vals: values, ndarray of shape N

    Returns
    -------
        q: equidistantly spaced log10(iters[0]) to log10(iters[-1])
        curv: curvature of log10(vals) interpolated onto q
    """
    from scipy.interpolate import InterpolatedUnivariateSpline
    if iters[0] == 0:
        raise ValueError("First iters value must not be zero.")
    if not isinstance(length, int): length = len(vals)
    q = np.linspace(np.log10(iters[0]), np.log10(iters[-1]), length)
    dq = q[1] - q[0]
    spline_q = InterpolatedUnivariateSpline(np.log10(iters), np.log10(vals))
    qv = spline_q(q)

    first_derivative = np.gradient(qv, dq)
    snd_derivative = np.gradient(first_derivative, dq)
    curv = snd_derivative / (1 + first_derivative ** 2) ** 1.5
    return q, curv


def calc_curv(vals):
    """ curvature of vals under assumption that x-axis satisfies dx=1
    """
    first_derivative = np.gradient(vals)
    snd_derivative = np.gradient(first_derivative)
    curv = snd_derivative / (1 + first_derivative ** 2) ** 1.5
    return curv


def calc_chi2(Sh, M, bd, delta):
    """ Chi^2 values
    """
    if len(Sh.shape) == 2:
        chi2 = np.zeros(Sh.shape[0])
        for k in range(Sh.shape[0]):
            Mhat = np.convolve(Sh[k, :], bd, 'same')
            Mhat = Mhat / np.trapz(Mhat, dx=delta)
            chi2[k] = np.sum((M - Mhat) ** 2 / Mhat)
    else:
        Mhat = np.convolve(Sh, bd, 'same')
        Mhat = Mhat / np.trapz(Mhat, dx=delta)
        chi2 = np.sum((M - Mhat) ** 2 / Mhat)
    return chi2


class curv_progress(stopping_rule):
    """
    Calculate the root-mean-squared progress in the RichLucy iterations.
    Determine the optimal number of iterations from the curvature of the rms values.

    This criterion is published in:

    Eichstaedt, Schmaehling, Wuebbeler, Anhalt, Buenger, Krueger & Elster
    *Comparison of the Richardson-Lucy method and a classical approach for
    spectrometer bandwidth correction*, Metrologia vol. 50, 2013 `download paper <http://dx.doi.org/10.1088/0026-1394/50/2/107>`_
    """

    name = "curv(progress)"
    miniter = 10  # min. number of iterations for curv calculations
    orig_mode = True  # orig_mode = use global maximum

    RMSprogr = None
    curv = None
    q = None

    def __init__(self, M, btilde, noise,delta):
        self.Meas = M
        self.btilde = btilde
        self.noise = noise
        self.N = len(M)
        self.delta = delta

    def calc_stoppoint(self, SH):
        from scipy.signal import argrelmax

        lenSH, lM = np.shape(SH)

        self.maxiter = lenSH - 1

        self.output('\nCalculating optimal stopping iteration using curvature information\n')

        RMS = np.zeros(self.maxiter)
        self.output('Calculating root-mean-squared value of change from iteration to iteration..')
        for r in range(1, self.maxiter + 1):
            RMS[r - 1] = np.sqrt(np.mean((SH[r, :] - SH[r - 1, :]) ** 2))
            if np.mod(r, self.maxiter * 0.1) == 0:
                self.output('.')
        self.output('done.\n')

        self.RMSprogr = RMS[:]

        if self.maxiter > self.miniter and self.maxiter > self.min_auto_iter:
            if any(np.isnan(RMS)) or any(np.isinf(RMS)) or np.min(RMS) == np.max(RMS):
                self.output("Calculation of automatic stopping by max curvature method failed.\n")
                return -1

            self.output('Calculating curvature of rms values.\n')
            iters = np.arange(1, self.maxiter + 1)
            q, curv = calc_log_curv(iters, RMS)
            qmin = np.nonzero(q >= np.log10(self.miniter))[0][0]
            indopt = 1
            if self.orig_mode:  # global maximum
                qopt = q[curv[qmin:].argmax() + qmin]
                indopt = np.round(10 ** qopt)
            else:
                qmax = argrelmax(curv[:-1])[0]
                relevant_qmax = []
                for qm in qmax:
                    if qm >= qmin:
                        relevant_qmax.append(qm)
                if len(relevant_qmax) > 0:
                    indopts = []
                    cmax = curv[relevant_qmax]
                    inds = cmax.argsort()[-3:]  # take the largest 3 local maxima
                    for ind in inds:
                        qopt = q[curv == curv[relevant_qmax[ind]]]
                        qopt = qopt[0]
                        # transform q-value to iteration number
                        iopt = np.flatnonzero(np.log10(iters) <= qopt)
                        indopts.append(iters[iopt[-1]])
                    # calculate error of estimated measurement for the candidates
                    self.output("potential stopping points are:\n " + repr(indopts) + "\n")
                    chi2 = calc_chi2(SH[indopts, :], self.Meas, self.btilde, self.delta)
                    indopt = indopts[chi2.argmin()]
            self.crit_measure = curv[:]
            self.curv = curv[:]
            self.q = q
        else:
            self.output(
                'Warning: Maximum number of iterations is too small for calculation of curvature. Automatic stopping not available.\n')
            indopt = -1

        if indopt < 0:
            self.output('Calculation of optimal stopping using the max curvature method failed.\n')
        else:
            self.output('Estimated optimal stopping point is iter:' + repr(indopt) + '\n')
        self.output('\n')
        self.indopt = indopt
        return indopt

    def plot(self, fig_nr=1):
        from matplotlib.pyplot import figure, clf, semilogx, legend, draw
        figure(fig_nr)
        clf()
        semilogx(range(1, self.maxiter + 1), np.log10(self.RMSprogr), label=r"$\log_{10}$ of progress")
        semilogx(10 ** self.q, self.curv, label="curvature")
        legend(loc="best")
        draw()


class curv_entropy(stopping_rule):
    """
    For each iteration calculate the correction deltaH and the max. entropy value deltaS.
    A flat line is used as reference spectrum.
    Use the curvature of the evolution of deltaH vs deltaS to determine optimal stopping.

    This criterion was published by L.B. Lucy:

    L.B. Lucy, 'Optimum strategies for inverse problems in statistical astronomy'
    Astron. Astrophys. 289, 983-994, 1994

    """

    chi = 0.08
    alpha = 0.1
    miniter = 20
    name = "curv(ent)"

    def __init__(self, M, bdtilde, noise, delta):
        self.Meas = M
        self.bdtilde = bdtilde
        self.noise = noise
        self.delta = delta

    def entropy(self, phat):
        ph = phat / np.trapz(phat, dx=self.delta)
        inds = np.nonzero(ph > 1e-12)
        return -np.sum(ph[inds] * np.log(ph[inds] / self.chi))

    def DeltaS(self, phat):
        ph = phat / np.trapz(phat, dx=self.delta)
        ph[ph < 1e-16] = 1e-12
        return -self.alpha * ph * (self.entropy(phat) + np.log(ph / self.chi))

    def DeltaH(self, phat):
        ph = phat / np.trapz(phat, dx=self.delta)
        Mm = self.Meas / np.trapz(self.Meas, dx=self.delta)
        Mhat = np.convolve(ph, self.bdtilde, 'same')
        return ph * (np.convolve(Mm / Mhat, self.bdtilde[::-1], 'same'))

    def curv(self, T1, T2):
        dT1 = np.gradient(T1)
        dT2 = np.gradient(T2)
        ddT1 = np.gradient(dT1)
        ddT2 = np.gradient(dT2)
        return abs(dT1 * ddT2 - ddT1 * dT2) / (dT1 ** 2 + dT2 ** 2) ** 1.5

    def calc_stoppoint(self, SH):
        self.output("Calculating optimal stopping point using entropy curvature\n")
        self.maxiter = SH.shape[0] - 1
        dS = np.zeros(SH.shape[0] - 1)
        dH = np.zeros_like(dS)

        for k in range(1, SH.shape[0]):
            dH[k - 1] = np.max(np.abs((self.DeltaH(SH[k, :]))))
            dS[k - 1] = np.max(np.abs(self.DeltaS(SH[k, :])))

        SHcurv = self.curv(dS, dH)
        indopt = np.argmax(SHcurv[self.miniter:]) + self.miniter
        self.crit_measure = SHcurv[:]
        self.indopt = indopt

        self.output("point of maximum curvature is at iteration %d\n" % indopt)
        return indopt

    def plot(self, fig_nr=1):
        from matplotlib.pyplot import figure, clf, semilogx, draw, xlabel, ylabel, axvline
        figure(fig_nr)
        clf()
        semilogx(range(1, self.maxiter + 1), np.log10(self.crit_measure), label=r"curv. of entropy")
        xlabel("iteration")
        ylabel("curvature of entropy")
        axvline(self.indopt, color="k", linestyle="--")
        draw()

def RichLucy(b, delta, M, maxiter=500, autostop=True, display=True, stoprule=None, returnAll=False, initialShat=None):
    """
    Richardson Lucy iteration under the assumption that the sampling of bandpass function b and measured spectrum M is
    equidistant and that the wavelength step size of b and M is equal.

    Parameters
    ----------
        b: ndarray of shape (N,) with the bandpass function values
        delta: float, wavelength step size for b and M
        M: ndarray of shape (L,) with the measured spectrum values
        maxiter: (optional) maximum number of iterations, default is 500
        autostop: (optional) boolean whether to automatically find optimal iteration number (default: True)
        display: (optional) boolean whether to print information to the command line (default: True)
        stoprule: (optional) a stopping_rule object to calculate the optimal iteration number (default: None)
        returnAll: (optional) boolean whether to return all intermediate estimates (default: False)
        initialShat: (optional) ndarray of shape (L,) with an initial estimate of the input spectrum (default: None)

    Returns
    -------
        Shat, SH: ndarray of shape (L,) - the estimated spectrum and (if returnAll=True) ndarray of shape(maxiter+1,L) of all intermediate results
    """

    def output(text):
        if display:
            sys.stdout.write(text)

    output('\n -------------- Richardson-Lucy algorithm version  --------------\n')

    if np.count_nonzero(M < 0) > 0 or np.count_nonzero(b < 0) > 0:
        raise ValueError("Measured spectrum and bandpass function must not have negative values.")

    if np.abs(np.trapz(b, dx=delta) - 1) > 1e-4:
        raise ValueError("Line-spread function must be normalized.")

    if issubclass(type(stoprule), stopping_rule):
        calc_stop = True
        stoprule.display = display
        if maxiter < stoprule.min_auto_iter:
            output('Warning: Maximum number of iterations is too small. Automatic stopping not available.\n')
            autostop = False
    else:
        output("No stopping rule defined. Taking maximum number of iterations instead.\n")
        calc_stop = False
        autostop = False

    # transform continuous b to discrete (impulse invariance technique)
    bd = b * delta
    bdtilde = bd[::-1]

    # adjust length of M and b for proper convolution
    if M.size < b.size:
        sys.stdout.write(
            "The bandpass function seems to be defined over a larger wavelength region than the actual measurement.\n")
        sys.stdout.write("Padding the missing spectrum values with zeros, but results may be inaccurate\n")
        leng_diff = b.size - M.size
        M = np.concatenate((np.zeros(leng_diff), M, np.zeros(leng_diff)))
        padded = True
    else:
        leng_diff = 0
        padded = False

    # allocate computer memory
    if autostop or calc_stop or returnAll:
        saveAll = True
    else:
        saveAll = False

    if saveAll:
        SH = np.zeros((maxiter + 1, M.size))
    else:
        SH = np.zeros(M.size)

    # initial 'iteration'
    r = 0
    if isinstance(initialShat, np.ndarray):
        if len(initialShat) == M.size:
            if saveAll:
                SH[r, :] = initialShat
            else:
                SH = initialShat
        else:
            output("User-supplied initial estimate does not match measured spectrum and will be ignored.\n")
            if saveAll:
                SH[r, :] = M[:]
            else:
                SH = M[:]
    else:
        if saveAll:
            SH[r, :] = M[:]
        else:
            SH = M[:]

            # additional iterations
    if autostop:
        output('Richardson-Lucy method calculating optimal stopping point using ' + stoprule.name + "\n")
        output('Step 1: Carry out ' + repr(int(maxiter)) + ' iterations.\n')
    else:
        output('Richardson-Lucy method with ' + repr(int(maxiter)) + ' iterations\n')
    #######################    original RL iterations
    while r < maxiter:
        r = r + 1
        # actual RichLucy step
        if saveAll:
            tmp1 = np.convolve(SH[r - 1, :], bdtilde, 'same')
        else:
            tmp1 = np.convolve(SH, bdtilde, 'same')
        tmp1[tmp1 != 0] = M[tmp1 != 0] / tmp1[tmp1 != 0]

        # if anything went wrong during convolution - set to zero
        count_all = np.count_nonzero(tmp1 == 0)
        tmp1[np.isnan(tmp1)] = 0
        tmp1[np.isinf(tmp1)] = 0
        count_bad = np.count_nonzero(tmp1 == 0) - count_all
        if display and count_bad > 0:
            sys.stdout.write(
                'After convolution in RL iteration %d, %d estimated values were set to zero. \n' % (r, count_bad))

        tmp2 = np.convolve(tmp1, bdtilde[::-1], 'same')
        if saveAll:
            Shat_new = SH[r - 1, :] * tmp2
        else:
            Shat_new = SH * tmp2

        Shat_new[np.isnan(Shat_new)] = 0
        Shat_new[np.isinf(Shat_new)] = 0
        if saveAll:
            SH[r, :] = Shat_new
        else:
            SH = Shat_new
        if np.mod(r, maxiter * 0.05) == 0 and display:
            sys.stdout.write(".")
            ######################
    if padded:
        if saveAll:
            SH = SH[:, leng_diff:-leng_diff]
        else:
            SH = SH[leng_diff:-leng_diff]

    output(' Done.\n')
    if autostop:
        output('Step 2: Calculating optimal stopping point\n')

    if calc_stop:
        indopt = stoprule.calc_stoppoint(SH)
    else:
        indopt = maxiter

    if indopt < 0 and autostop:
        output('Warning: Calculation of optimal stopping failed. Using measured spectrum as best estimate.\n')
        Shat = M[:]
    else:
        if autostop:
            if display:
                print('Optimal number of iterations = ' + repr(int(indopt) + 1))
            Shat = np.squeeze(SH[indopt - 1, :])
        else:
            if saveAll:
                Shat = SH[-1, :]
            else:
                Shat = SH

    if returnAll:
        return Shat, SH
    else:
        return Shat
