# -*- coding: utf-8 -*-
"""
The :mod:`PyDynamic.identification` module implements the least-squares fit of an IIR or FIR digital filter
to a given frequency response.
"""

from .fit_filter import LSFIR, LSIIR

__all__ = ['LSFIR', 'LSIIR']