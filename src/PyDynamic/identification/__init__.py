# -*- coding: utf-8 -*-
"""
.. deprecated:: 2.0.0
    The module *identification* is combined with the module *deconvolution* and
    renamed to :mod:`PyDynamic.model_estimation` since the last major release 2.0.0.
    *identification* might be removed any time. Please switch to the current module
    immediately.
"""

from .fit_filter import LSFIR, LSIIR
from .fit_transfer import fit_sos

__all__ = ["LSFIR", "LSIIR", "fit_sos"]
