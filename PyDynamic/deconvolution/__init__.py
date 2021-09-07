# -*- coding: utf-8 -*-
"""
The :doc:`PyDynamic.deconvolution` package implements methods for the design of
digital deconvolution filters by least-squares fitting to the reciprocal of a given
frequency response with associated uncertainties.

.. seealso::

   - `initial project website <https://www.euramet.org/research-innovation/search
     -research-projects/details/project/standards-and-software-to-maximise-end-user
     -uptake-of-nmi-calibrations-of-dynamic-force-torque-and/>`_
   - `GitHub website <https://www.github.com/PTB-M4D/PyDynamic>`_
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
