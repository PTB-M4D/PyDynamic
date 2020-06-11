# -*- coding: utf-8 -*-
"""
The package :doc:`PyDynamic.model_estimation` implements methods for the model
estimation by least-squares fitting and the identification of transfer function models
with associated uncertainties.
"""

# See http://mathmet.org/projects/14SIP08 and
# https://www.github.com/PTB-PSt1/PyDynamic

from .fit_filter import (LSFIR, LSIIR, invLSFIR, invLSFIR_unc, invLSFIR_uncMC,
                         invLSIIR, invLSIIR_unc)
from .fit_transfer import fit_som

__all__ = [
    "LSFIR",
    "LSIIR",
    "invLSFIR",
    "invLSIIR",
    "invLSFIR_unc",
    "invLSFIR_uncMC",
    "invLSIIR_unc",
    "fit_som",
]
