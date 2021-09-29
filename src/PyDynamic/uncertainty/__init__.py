# -*- coding: utf-8 -*-
"""
The :mod:`PyDynamic.uncertainty` module implements methods for the
propagation of uncertainty for the most common use cases in the analysis of
dynamic measurements including filtering, Monte Carlo methods and time
series interpolation.

.. seealso::

   `initial project website <https://www.euramet.org/research-innovation/search
   -research-projects/details/project/standards-and-software-to-maximise-end-user
   -uptake-of-nmi-calibrations-of-dynamic-force-torque-and/>`_
   `GitHub website <https://www.github.com/PTB-M4D/PyDynamic>`_
"""

from .interpolate import interp1d_unc
from .propagate_DFT import (
    AmpPhase2DFT,
    AmpPhase2Time,
    DFT2AmpPhase,
    DFT_deconv,
    DFT_multiply,
    GUM_DFT,
    GUM_iDFT,
    Time2AmpPhase,
)
from .propagate_filter import FIRuncFilter, IIRuncFilter
from .propagate_MonteCarlo import MC, SMC, UMC, UMC_generic
from ..misc.noise import ARMA

__all__ = [
    "ARMA",
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
