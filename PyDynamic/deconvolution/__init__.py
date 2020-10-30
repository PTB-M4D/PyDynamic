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

from .fit_filter import (
    LSFIR,
    LSIIR,
    LSFIR_unc,
    LSFIR_uncMC,
    LSIIR_unc,
)

__all__ = [
    "LSFIR",
    "LSIIR",
    "LSFIR_unc",
    "LSFIR_uncMC",
    "LSIIR_unc",
]
