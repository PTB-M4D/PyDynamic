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


def LSFIR(*args, **kwargs):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSFIR`"""
    raise DeprecationWarning(f"{warning}:func:`PyDynamic.model_estimation.invLSFIR`.")


def LSFIR_unc(*args, **kwargs):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSFIR_unc`
    """
    raise DeprecationWarning(
        f"{warning}:func:`PyDynamic.model_estimation.invLSFIR_unc`."
    )


def LSFIR_uncMC(*args, **kwargs):
    """
    .. deprecated:: 1.4.1
        Please use :func:`PyDynamic.model_estimation.invLSFIR_uncMC`
    """
    raise DeprecationWarning(
        f"{warning}:func:`PyDynamic.model_estimation.invLSFIR_uncMC`."
    )


def LSIIR(*args, **kwargs):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSIIR`"""
    raise DeprecationWarning(f"{warning}:func:`PyDynamic.model_estimation.invLSIIR`.")


def LSIIR_unc(*args, **kwargs):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.invLSIIR_unc`
    """
    raise DeprecationWarning(
        f"{warning}:func:`PyDynamic.model_estimation.invLSIIR_unc`."
    )
