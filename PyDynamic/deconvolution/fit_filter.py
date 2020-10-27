# -*- coding: utf-8 -*-
"""
.. deprecated:: 2.0.0
    The module *deconvolution* is combined with the module *identification* and
    renamed to :mod:`PyDynamic.model_estimation` since the last major release 2.0.0.
    *deconvolution* might be removed any time. Please switch to the current module
    immediately. The previously known functions `LSFIR()`, `LSFIR_unc()`, `LSIIR()`,
    `LSIIR_unc()`, `LSFIR_uncMC()` were prefixed with an "inv" for "inverse",
    indicating the treatment of the reciprocal of frequency response values. Please
    use the new function names (e.g.
    :func:`PyDynamic.model_estimation.fit_filter.invLSIIR_unc`).
"""

import warnings


warning = (
    "The module *deconvolution* is combined with the module *identification* "
    "and renamed to :mod:`PyDynamic.model_estimation` since the last major "
    "release 2.0.0. *deconvolution* might be removed any time. Please switch"
    " to the current module immediately. The previously known functions `LSFIR("
    ")`, `LSFIR_unc()`, `LSIIR()`, `LSIIR_unc()`, `LSFIR_uncMC()` were"
    " prefixed with an 'inv' for 'inverse', indicating the treatment of the "
    "reciprocal of frequency response values. Please use the current function "
    "name "
)

__all__ = [
    "LSFIR",
    "LSIIR",
    "LSFIR_unc",
    "LSFIR_uncMC",
    "LSIIR_unc",
]


def LSFIR(**kwargs):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSFIR`"""
    warnings.warn(
        f"{warning}:func:`PyDynamic.model_estimation.invLSFIR`.", DeprecationWarning
    )


def LSFIR_unc(**kwargs):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSFIR_unc`
    """
    warnings.warn(
        f"{warning}:func:`PyDynamic.model_estimation.invLSFIR_unc`.", DeprecationWarning
    )


def LSFIR_uncMC(**kwargs):
    """
    .. deprecated:: 1.4.1
        Please use :func:`PyDynamic.model_estimation.invLSFIR_uncMC`
    """
    warnings.warn(
        f"{warning}:func:`PyDynamic.model_estimation.invLSFIR_uncMC`.",
        DeprecationWarning,
    )


def LSIIR(**kwargs):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSIIR`"""
    warnings.warn(
        f"{warning}:func:`PyDynamic.model_estimation.invLSIIR`.", DeprecationWarning
    )


def LSIIR_unc(**kwargs):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSIIR_unc`
    """
    warnings.warn(
        f"{warning}:func:`PyDynamic.model_estimation.invLSIIR_unc`.", DeprecationWarning
    )
