# -*- coding: utf-8 -*-
"""
The package :doc:`PyDynamic.model_estimation` implements methods for the model
estimation by least-squares fitting and the identification of transfer function models
with associated uncertainties.
"""

# See http://mathmet.org/projects/14SIP08 and
# https://www.github.com/PTB-PSt1/PyDynamic

from .fit_filter import (LSFIR, LSIIR, iLSFIR, iLSFIR_unc, iLSFIR_uncMC,
                         iLSIIR, iLSIIR_unc)
from .fit_transfer import fit_sos

__all__ = [
    "LSFIR",
    "LSIIR",
    "iLSFIR",
    "iLSIIR",
    "iLSFIR_unc",
    "iLSFIR_uncMC",
    "iLSIIR_unc",
    "fit_sos",
]
