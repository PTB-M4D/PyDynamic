# -*- coding: utf-8 -*-
"""
The :mod:`PyDynamic.identification` package implements the least-squares fit of an
IIR or FIR digital filter to a given frequency response and the identification of
transfer function models.

.. seealso::

   - `initial project website <https://www.euramet.org/research-innovation/search
     -research-projects/details/project/standards-and-software-to-maximise-end-user
     -uptake-of-nmi-calibrations-of-dynamic-force-torque-and/>`_
   - `GitHub website <https://www.github.com/PTB-M4D/PyDynamic>`_
"""

from .fit_filter import LSFIR, LSIIR
from .fit_transfer import fit_sos

__all__ = ["LSFIR", "LSIIR", "fit_sos"]
