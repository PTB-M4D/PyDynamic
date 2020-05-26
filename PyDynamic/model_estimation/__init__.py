# -*- coding: utf-8 -*-
"""
The :doc:`PyDynamic.model_estimation` package implements methods for the model
estimation by least-squares fitting and the identification of transfer function models
with associated uncertainties.
"""

# See http://mathmet.org/projects/14SIP08 and
# https://www.github.com/eichstaedtPTB/PyDynamic

from .fit_filter import (
    LSFIR,
    LSIIR,
    iLSFIR,
    iLSIIR,
    iLSFIR_unc,
    iLSFIR_uncMC,
    iLSIIR_unc,
)

__all__ = [
    "LSFIR",
    "LSIIR",
    "iLSFIR",
    "iLSIIR",
    "iLSFIR_unc",
    "iLSFIR_uncMC",
    "iLSIIR_unc",
]
