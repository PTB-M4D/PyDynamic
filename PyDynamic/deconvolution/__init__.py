# -*- coding: utf-8 -*-
"""
The :doc:`PyDynamic.deconvolution` package implements methods for the design of
digital deconvolution filters by least-squares fitting to the reciprocal of a given
frequency response with associated uncertainties.
"""

# See http://mathmet.org/projects/14SIP08 and
# https://www.github.com/eichstaedtPTB/PyDynamic

from .fit_filter import (
    invLSFIR,
    invLSIIR,
    invLSFIR_unc,
    invLSFIR_uncMC,
    invLSIIR_unc,
    LSFIR,
    LSIIR,
    LSFIR_unc,
    LSFIR_uncMC,
    LSIIR_unc,
)

__all__ = [
    "invLSFIR",
    "invLSIIR",
    "invLSFIR_unc",
    "invLSFIR_uncMC",
    "invLSIIR_unc",
    "LSFIR",
    "LSIIR",
    "LSFIR_unc",
    "LSFIR_uncMC",
    "LSIIR_unc",
]
