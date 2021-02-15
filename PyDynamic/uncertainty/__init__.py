# -*- coding: utf-8 -*-
"""
The :mod:`PyDynamic.uncertainty` module implements methods for the
propagation of uncertainty for the most common use cases in the analysis of
dynamic measurements including filtering, Monte Carlo methods and time
series interpolation.

# See http://mathmet.org/projects/14SIP08 and
# https://www.github.com/eichstaedtPTB/PyDynamic
"""

from .propagate_DFT import (
    GUM_DFT,
    GUM_iDFT,
    DFT_deconv,
    DFT_multiply,
    DFT2AmpPhase,
    AmpPhase2DFT,
    AmpPhase2Time,
    Time2AmpPhase,
)

from .propagate_filter import FIRuncFilter, IIRuncFilter

from .propagate_MonteCarlo import MC, SMC, UMC, UMC_generic

from .interpolation import interp1d_unc

__all__ = [
    "GUM_DFT",
    "GUM_iDFT",
    "DFT_deconv",
    "DFT_multiply",
    "DFT2AmpPhase",
    "AmpPhase2DFT",
    "AmpPhase2Time",
    "Time2AmpPhase",
    "FIRuncFilter",
    "IIRuncFilter",
    "MC",
    "SMC",
    "UMC",
    "UMC_generic",
    "interp1d_unc",
]
