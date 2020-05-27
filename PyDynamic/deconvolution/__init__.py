# -*- coding: utf-8 -*-
"""
The :doc:`PyDynamic.deconvolution` package implements methods for the design of
digital deconvolution filters by least-squares fitting to the reciprocal of a given
frequency response with associated uncertainties.
"""

# See https://mathmet.org/projects/14SIP08 and
# https://www.github.com/PTB-PSt1/PyDynamic

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
