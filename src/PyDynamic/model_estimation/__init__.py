"""Fit IIR and FIR filters and identify transfer function models with uncertainties

The package :doc:`PyDynamic.model_estimation` implements methods for the model
estimation by least-squares fitting and the identification of transfer function models
with associated uncertainties.

.. seealso::

   - `initial project website <https://www.euramet.org/research-innovation/search
     -research-projects/details/project/standards-and-software-to-maximise-end-user
     -uptake-of-nmi-calibrations-of-dynamic-force-torque-and/>`_
   - `GitHub website <https://www.github.com/PTB-M4D/PyDynamic>`_
"""
__all__ = ["fit_som", "LSFIR", "LSIIR"]

from .fit_filter import LSFIR, LSIIR
from .fit_transfer import fit_som
