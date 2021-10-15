"""This module contains a function for the identification of transfer function models:

* :func:`fit_som`: Fit second-order model to complex-valued frequency response
"""
from typing import Optional, Union

import numpy as np

from PyDynamic.misc.tools import (
    is_2d_matrix,
    is_2d_square_matrix,
    number_of_rows_equals_vector_dim,
    progress_bar,
)

__all__ = ["fit_som"]


def fit_som(
    f: np.ndarray,
    H: np.ndarray,
    UH: Optional[np.ndarray] = None,
    weighting: Optional[np.ndarray] = None,
    MCruns: Optional[Union[int, None]] = 10000,
    scaling: Optional[float] = 1e-3,
    verbose: Optional[bool] = False,
):
    """Fit second-order model to complex-valued frequency response

    Fit second-order model (spring-damper model) with parameters
    :math:`S_0, delta` and :math:`f_0` to complex-valued frequency response
    with uncertainty associated with real and imaginary parts.

    For a transformation of an uncertainty associated with amplitude and
    phase to one associated with real and imaginary parts,
    see :mod:`PyDynamic.uncertainty.propagate_DFT.AmpPhase2DFT`.

    Parameters
    ----------
    f : (M,) np.ndarray
        vector of frequencies
    H : (2M,) np.ndarray
        real and imaginary parts of measured frequency response values at
        frequencies f
    UH : (2M,) or (2M,2M) np.ndarray, optional
        uncertainties associated with real and imaginary parts
        When UH is one-dimensional, it is assumed to contain standard
        uncertainties; otherwise it is taken as covariance matrix. When UH is not
        specified no uncertainties assoc. with the fit are calculated, which is the
        default behaviour.
    weighting : str or (2M,) np.ndarray, optional
        Type of weighting associated with frequency responses, can be ('diag',
        'cov') if UH is given, or Numpy array of weights, defaults to None,
        which means all values are considered equally important
    MCruns : int, optional
        Number of Monte Carlo trials for propagation of uncertainties, defaults to
        10000. When MCruns is set to 'None', matrix multiplication is used for the
        propagation of uncertainties. However, in some cases this can cause trouble.
    scaling : float, optional
        scaling of least-squares design matrix for improved fit quality, defaults to
        1e-3
    verbose : bool, optional
        if True a progressbar will be printed to console during the Monte Carlo
        simulations, if False nothing will be printed out, defaults to False
    Returns
    -------
    p : np.ndarray
        vector of estimated model parameters [S0, delta, f0]
    Up : np.ndarray
        covariance associated with parameter estimate
    """
    n = len(f)
    two_n = len(H)
    if 2 * n != two_n:
        raise ValueError(
            "fit_som: vector H of real and imaginary parts of frequency "
            "responses is expected to contain exactly twice as many elements as "
            f"vector f of frequencies. Please adjust f, which has {n} "
            f"elements or H, which has {two_n} elements."
        )
    h_real = H[:n]
    h_imag = H[n:]
    h_complex = h_real + 1j * h_imag
    if MCruns is None or UH is None:
        if np.any(h_complex == 0):
            raise ValueError(
                "fit_som: if MCruns is None or UH is None, uncertainty "
                "propagation is done via matrix multiplication as proposed in "
                "GUM S2, so frequency responses must not be zero anywhere but H "
                f"equals {H}."
            )
    elif not isinstance(MCruns, int):
        raise ValueError(
            f"fit_som: MCruns is expected to be None or of type int, but MCruns is of "
            f"type {type(MCruns)}."
        )

    if isinstance(weighting, np.ndarray):
        if len(weighting) != two_n:
            raise ValueError(
                "fit_som: if weighting is provided as a vector, its number of elements "
                "is expected to match the number of elements in H but weighting has "
                f"{len(weighting)} elements and H has {two_n} elements."
            )
        W = np.diag(weighting)
    elif isinstance(weighting, str) and weighting not in {"diag", "cov"}:
        raise ValueError(
            "fit_som: if weighting is provided as string, it is expected to be either"
            f"'diag' or 'cov', but '{weighting}' was given."
        )
    else:
        W = np.eye(two_n)

    if UH is not None:
        if not isinstance(UH, np.ndarray):
            raise ValueError(
                "fit_som: if UH is provided, it is expected to be of type np.ndarray, "
                f"but UH is of type {type(UH)}."
            )
        if not number_of_rows_equals_vector_dim(matrix=UH, vector=H):
            raise ValueError(
                "fit_som: number of rows of UH and number of elements of H are "
                f"expected to match. But H has {len(H)} elements and UH is of shape "
                f"{UH.shape}."
            )
        if is_2d_matrix(UH):
            if not is_2d_square_matrix(UH):
                raise ValueError(
                    "fit_som: if UH is a matrix, it is expected to be square but UH "
                    f"is of shape {UH.shape}."
                )
            # propagate to real and imaginary parts of reciprocal using Monte Carlo
            h_mc = np.random.multivariate_normal(H, UH, MCruns)
            h_mc_complex = h_mc[:, :n] + 1j * h_mc[:, n:]

        else:  # is_vector(UH)
            h_mc_real = np.tile(h_real, (MCruns, 1)) + np.random.randn(
                MCruns, n
            ) * np.tile(UH[:n], (MCruns, 1))
            h_mc_imag = np.tile(h_imag, (MCruns, 1)) + np.random.randn(
                MCruns, n
            ) * np.tile(UH[n:], (MCruns, 1))
            h_mc_complex = h_mc_real + 1j * h_mc_imag

        inverted_h_mc = np.c_[np.real(1 / h_mc_complex), np.imag(1 / h_mc_complex)]
        covariance_inverted_h_mc = np.cov(inverted_h_mc, rowvar=False)

        if isinstance(weighting, str):
            if weighting == "diag":
                W = np.diag(np.diag(covariance_inverted_h_mc))
            else:  # weighting == "cov"
                W = covariance_inverted_h_mc

        # Apply GUM S2
        if MCruns is not None:
            # Monte Carlo
            MU = np.zeros((MCruns, 3))
            for i_monte_carlo_run in range(MCruns):
                iri = inverted_h_mc[i_monte_carlo_run, :]
                om = 2 * np.pi * f * scaling
                E = np.c_[np.ones(n), 2j * om, -(om ** 2)]
                X = np.r_[np.real(E), np.imag(E)]

                XVX = X.T.dot(np.linalg.solve(W, X))
                XVy = X.T.dot(np.linalg.solve(W, iri))

                MU[i_monte_carlo_run, :] = np.linalg.solve(XVX, XVy)

                if verbose:
                    progress_bar(
                        i_monte_carlo_run,
                        MCruns,
                        prefix="Monte Carlo for fit_som() running:",
                    )
            MU[:, 1] *= scaling
            MU[:, 2] *= scaling ** 2

            # Calculate S0, delta and f0
            PARS = np.c_[
                1 / MU[:, 0],
                MU[:, 1] / np.sqrt(np.abs(MU[:, 0] * MU[:, 2])),
                np.sqrt(np.abs(MU[:, 0] / MU[:, 2])) / 2 / np.pi,
            ]

            pars = PARS.mean(axis=0)
            Upars = np.cov(PARS, rowvar=False)
        else:  # MCruns == None, so apply GUM S2 linear propagation
            iri = np.r_[np.real(1 / h_complex), np.imag(1 / h_complex)]
            om = 2 * np.pi * f
            E = np.c_[np.ones(n), 2j * om, -(om ** 2)]
            X = np.r_[np.real(E), np.imag(E)]

            XVX = X.T.dot(np.linalg.solve(W, X))
            XVy = X.T.dot(np.linalg.solve(W, iri))

            mu = np.linalg.solve(XVX, XVy)
            iXVX = np.linalg.inv(XVX)
            XVUVX = np.dot(
                X.T.dot(np.linalg.solve(W, covariance_inverted_h_mc)),
                np.linalg.solve(W, X),
            )
            Umu = iXVX.dot(XVUVX).dot(iXVX)

            pars = np.r_[
                1 / mu[0],
                mu[1] / np.sqrt(np.abs(mu[0] * mu[2])),
                np.sqrt(np.abs(mu[0] / mu[2])) / 2 / np.pi,
            ]
            C = np.array(
                [
                    [-1 / mu[0] ** 2, 0, 0],
                    [
                        -mu[1] / (2 * (np.abs(mu[0] + mu[2])) ** (3 / 2)),
                        1 / np.sqrt(np.abs(mu[0] + mu[2])),
                        -mu[1] / (2 * np.abs(mu[0] + mu[2]) ** (3 / 2)),
                    ],
                    [
                        1 / (4 * np.pi * mu[2] * np.sqrt(np.abs(mu[0] / mu[2]))),
                        0,
                        -mu[0]
                        / (4 * np.pi * mu[2] ** 2 * np.sqrt(np.abs(mu[0] / mu[2]))),
                    ],
                ]
            )
            Upars = C.dot(Umu.dot(C.T))

        return pars, Upars
    # UH is None, so apply GUM S2 linear propagation
    iri = np.r_[np.real(1 / h_complex), np.imag(1 / h_complex)]
    om = 2 * np.pi * f * scaling
    E = np.c_[np.ones(n), 2j * om, -(om ** 2)]
    X = np.r_[np.real(E), np.imag(E)]

    XVX = X.T.dot(np.linalg.solve(W, X))
    XVy = X.T.dot(np.linalg.solve(W, iri))

    mu = np.linalg.solve(XVX, XVy)
    mu[1] *= scaling
    mu[2] *= scaling ** 2
    pars = np.r_[
        1 / mu[0],
        mu[1] / np.sqrt(np.abs(mu[0] * mu[2])),
        np.sqrt(np.abs(mu[0] / mu[2])) / 2 / np.pi,
    ]
    return pars
