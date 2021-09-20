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

from .propagate_filter import FIRuncFilter, IIRuncFilter, IIR_get_initial_state

from .propagate_MonteCarlo import MC, SMC, UMC, UMC_generic

from .interpolation import interp1d_unc

from .propagate_DWT import (
    dwt,
    wave_dec,
    wave_dec_realtime,
    inv_dwt,
    wave_rec,
    filter_design,
    dwt_max_level,
)

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
    "IIR_get_initial_state",
    "MC",
    "SMC",
    "UMC",
    "UMC_generic",
    "interp1d_unc",
    "dwt",
    "wave_dec",
    "wave_dec_realtime",
    "inv_dwt",
    "wave_rec",
    "filter_design",
    "dwt_max_level",
]