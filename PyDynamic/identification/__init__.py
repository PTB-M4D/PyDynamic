# -*- coding: utf-8 -*-
"""
The :mod:`PyDynamic.identification` module implements the least-squares fit of an IIR or FIR digital filter
to a given frequency response.

# See http://mathmet.org/projects/14SIP08 and
# https://www.github.com/eichstaedtPTB/PyDynamic
"""

from .fit_filter import LSFIR, LSIIR
from .fit_transfer import fit_sos

__all__ = ['LSFIR', 'LSIIR', 'fit_sos']