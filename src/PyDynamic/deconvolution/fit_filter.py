# -*- coding: utf-8 -*-
"""
.. deprecated:: 1.2.71
    The module *deconvolution* will be combined with the module *identification* and
    renamed to *model_estimation* in the next major release 2.0.0. From then on you
    should only use the new module :doc:`PyDynamic.model_estimation` instead. The
    functions :func:`LSFIR`, :func:`LSFIR_unc`, :func:`LSIIR`, :func:`LSIIR_unc`,
    :func:`LSFIR_uncMC` are then prefixed with an "inv" for "inverse", indicating the
    treatment of the reciprocal of frequency response values. Please use the new
    function names (e.g. :func:`PyDynamic.model_estimation.fit_filter.invLSIIR_unc`)
    starting from version 1.4.1. The old function names without preceding "inv" will
    only be preserved until the release prior to version 2.0.0.

The :mod:`PyDynamic.deconvolution.fit_filter` module implements methods for the
design of digital deconvolution filters by least-squares fitting to the reciprocal of
a given frequency response with associated uncertainties.

This module for now still contains the following functions:

* :func:`LSFIR`: Least-squares fit of a digital FIR filter to the reciprocal of a
  given frequency response.
* :func:`LSFIR_unc`: Design of FIR filter as fit to reciprocal of frequency response
  values with uncertainty
* :func:`LSFIR_uncMC`: Design of FIR filter as fit to reciprocal of frequency
  response values with uncertainty via Monte Carlo
* :func:`LSIIR`: Design of a stable IIR filter as fit to reciprocal of frequency
  response values
* :func:`LSIIR_unc`: Design of a stable IIR filter as fit to reciprocal of frequency
  response values with uncertainty
"""

import warnings

from ..model_estimation.fit_filter import (
    invLSFIR,
    invLSFIR_unc,
    invLSFIR_uncMC,
    invLSIIR,
    invLSIIR_unc,
)

__all__ = [
    "LSFIR",
    "LSFIR_unc",
    "LSIIR",
    "LSIIR_unc",
    "LSFIR_uncMC",
]

warnings.warn(
    "The module *deconvolution* will be combined with the module *identification* and"
    "renamed to *model_estimation* in the next major release 2.0.0. From then on you"
    "should only use the new module *model_estimation* instead. The functions"
    "'LSFIR()', 'LSFIR_unc()', 'LSIIR()', 'LSIIR_unc()', 'LSFIR_uncMC()' are then"
    "prefixed with an 'inv' for 'inverse', indicating the treatment of the reciprocal"
    "of frequency response values. Please use the new function names (e.g. "
    ":func:`PyDynamic.model_estimation.fit_filter.invLSIIR_unc`) starting from version"
    "1.4.1.",
    DeprecationWarning,
)


def LSFIR(H, N, tau, f, Fs, Wt=None):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSFIR`
    """
    return invLSFIR(H=H, N=N, tau=tau, f=f, Fs=Fs, Wt=Wt)


def LSFIR_unc(H, UH, N, tau, f, Fs, wt=None, verbose=True, trunc_svd_tol=None):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSFIR_unc`
    """
    return invLSFIR_unc(
        H=H,
        UH=UH,
        N=N,
        tau=tau,
        f=f,
        Fs=Fs,
        wt=wt,
        verbose=verbose,
        trunc_svd_tol=trunc_svd_tol,
    )


def LSFIR_uncMC(H, UH, N, tau, f, Fs, verbose=True):
    """
    .. deprecated:: 1.4.1
        Please use :func:`PyDynamic.model_estimation.invLSFIR_uncMC`
    """
    return invLSFIR_uncMC(H=H, UH=UH, N=N, tau=tau, f=f, Fs=Fs, verbose=verbose)


def LSIIR(Hvals, Nb, Na, f, Fs, tau, justFit=False, verbose=True):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSIIR`"""
    return invLSIIR(
        Hvals=Hvals, Nb=Nb, Na=Na, f=f, Fs=Fs, tau=tau, justFit=justFit, verbose=verbose
    )


def LSIIR_unc(H, UH, Nb, Na, f, Fs, tau=0):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSIIR_unc`
    """
    return invLSIIR_unc(H=H, UH=UH, Nb=Nb, Na=Na, f=f, Fs=Fs, tau=tau)
