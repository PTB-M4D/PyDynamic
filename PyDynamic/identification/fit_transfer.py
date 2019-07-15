# -*- coding: utf-8 -*-
""" Collection of methods for the identification of transfer function models

This module contains the following functions:

* *fit_sos*: Fit second-order model to complex-valued frequency response

"""

import numpy as np

__all__ = ['fit_sos']


def fit_sos(f, H, UH=None, weighting=None, MCruns=None, scaling=1e-3):
    """ Fit second-order model to complex-valued frequency response

    Fit second-order model (spring-damper model) with parameters
    :math:`S_0, delta` and :math:`f_0` to complex-valued frequency response
    with uncertainty associated with real and imaginary parts.

    For a transformation of an uncertainty associated with amplitude and
    phase to one associated with real and imaginary parts,
    see `mod`::PyDynamic.uncertainty.propagate_DFT.AmpPhase2DFT.

    Parameters
    ----------
        f: np.ndarray of shape (M,)
            vector of frequencies
        H: np.ndarray of shape (2M,)
            real and imaginary parts of measured frequency response values at
            frequencies f
        UH: np.ndarray of shape (2M,) or (2M,2M)
            uncertainties associated with real and imaginary parts
            When UH is one-dimensional, it is assumed to contain standard
            uncertainties; otherwise it
            is taken as covariance matrix. When UH is not specified no
            uncertainties assoc. with the fit are calculated.
        weighting: str or array
            Type of weighting (None, 'diag', 'cov') or array of weights (
            length two times of f)
        MCruns: int, optional
            Number of Monte Carlo trials for propagation of uncertainties.
            When MCruns is 'None', matrix multiplication
            is used for the propagation of uncertainties. However, in some
            cases this can cause trouble.
        scaling: float
            scaling of least-squares design matrix for improved fit quality
    Returns
    -------
        p: np.ndarray
            vector of estimated model parameters [S0, delta, f0]
        Up: np.ndarray
            covariance associated with parameter estimate
    """
    assert (2 * len(f) == len(H))
    Hr = H[:len(f)]
    Hi = H[len(f):]

    if isinstance(UH, np.ndarray):
        assert (UH.shape[0] == 2 * len(f))
        if len(UH.shape) == 2:
            assert (UH.shape[0] == UH.shape[1])

        # propagate to real and imaginary parts of reciprocal using Monte Carlo

        if isinstance(MCruns, int) or isinstance(MCruns, float):
            runs = int(MCruns)
        else:
            runs = 10000
        if len(UH.shape) == 1:
            HR = np.tile(Hr, (runs, 1)) + \
                np.random.randn(runs, len(f)) * np.tile(UH[:len(f)], (runs, 1))
            HI = np.tile(Hi, (runs, 1)) + \
                np.random.randn(runs, len(f)) * np.tile(UH[len(f):], (runs, 1))
            HMC = HR + 1j * HI
        else:
            HRI = np.random.multivariate_normal(H, UH, runs)
            HMC = HRI[:, :len(f)] + 1j * HRI[:, len(f):]

        iRI = np.c_[np.real(1 / HMC), np.imag(1 / HMC)]
        iURI = np.cov(iRI, rowvar=False)

    if isinstance(weighting, str):
        if weighting == "diag":
            W = np.diag(np.diag(iURI))
        elif weighting == "cov":
            W = iURI
        else:
            print("Warning: Specified wrong type of weighting.")
            W = np.eye(2 * len(f))
    elif isinstance(weighting, np.ndarray):
        assert (len(weighting) == 2 * len(f))
        W = np.diag(weighting)
    else:
        W = np.eye(2 * len(f))

    if isinstance(UH, np.ndarray):
        # Apply GUM S2
        if isinstance(MCruns, int) or isinstance(MCruns, float):
            # Monte Carlo
            runs = int(MCruns)
            MU = np.zeros((runs, 3))
            for k in range(runs):
                iri = iRI[k, :]
                n = len(f)
                om = 2 * np.pi * f * scaling
                E = np.c_[np.ones(n), 2j * om, - om ** 2]
                X = np.r_[np.real(E), np.imag(E)]

                XVX = X.T.dot(np.linalg.solve(W, X))
                XVy = X.T.dot(np.linalg.solve(W, iri))

                MU[k, :] = np.linalg.solve(XVX, XVy)
            MU[:, 1] *= scaling
            MU[:, 2] *= scaling ** 2

            # Calculate S0, delta and f0
            PARS = np.c_[1 / MU[:, 0], MU[:, 1] / np.sqrt(
                np.abs(MU[:, 0] * MU[:, 2])), np.sqrt(
                np.abs(MU[:, 0] / MU[:, 2])) / 2 / np.pi]

            pars = PARS.mean(axis=0)
            Upars = np.cov(PARS, rowvar=False)
        else:  # apply GUM S2 linear propagation
            Hc = Hr + 1j * Hi
            assert (np.min(np.abs(Hc) > 0)),\
                "Frequency response cannot be equal to zero for inversion."
            iri = np.r_[np.real(1 / Hc), np.imag(1 / Hc)]
            n = len(f)
            om = 2 * np.pi * f
            E = np.c_[np.ones(n), 2j * om, - om ** 2]
            X = np.r_[np.real(E), np.imag(E)]

            XVX = X.T.dot(np.linalg.solve(W, X))
            XVy = X.T.dot(np.linalg.solve(W, iri))

            mu = np.linalg.solve(XVX, XVy)
            iXVX = np.linalg.inv(XVX)
            XVUVX = np.dot(X.T.dot(np.linalg.solve(W, iURI)),
                           np.linalg.solve(W, X))
            Umu = iXVX.dot(XVUVX).dot(iXVX)

            pars = np.r_[
                1 / mu[0], mu[1] / np.sqrt(np.abs(mu[0] * mu[2])), np.sqrt(
                    np.abs(mu[0] / mu[2])) / 2 / np.pi]
            C = np.array([[-1 / mu[0] ** 2, 0, 0],
                          [-mu[1] / (2 * (np.abs(mu[0] + mu[2])) ** (3 / 2)),
                           1 / np.sqrt(np.abs(mu[0] + mu[2])),
                           -mu[1] / (2 * np.abs(mu[0] + mu[2]) ** (3 / 2))],
                          [1 / (4 * np.pi * mu[2] * np.sqrt(
                              np.abs(mu[0] / mu[2]))), 0, -mu[0] / (
                                       4 * np.pi * mu[2] ** 2 *
                                       np.sqrt(np.abs(mu[0] / mu[2])))]])
            Upars = C.dot(Umu.dot(C.T))

        return pars, Upars
    else:
        Hc = Hr + 1j * Hi
        assert (np.min(np.abs(Hc)) > 0),\
            "Frequency response cannot be equal to zero for inversion."
        iri = np.r_[np.real(1 / Hc), np.imag(1 / Hc)]
        n = len(f)
        om = 2 * np.pi * f * scaling
        E = np.c_[np.ones(n), 2j * om, - om ** 2]
        X = np.r_[np.real(E), np.imag(E)]

        XVX = X.T.dot(np.linalg.solve(W, X))
        XVy = X.T.dot(np.linalg.solve(W, iri))

        mu = np.linalg.solve(XVX, XVy)
        mu[1] *= scaling
        mu[2] *= scaling ** 2
        pars = np.r_[1 / mu[0], mu[1] / np.sqrt(np.abs(mu[0] * mu[2])),
                     np.sqrt(np.abs(mu[0] / mu[2])) / 2 / np.pi]
        return pars
