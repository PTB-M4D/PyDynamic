# -*- coding: utf-8 -*-
"""
The :mod:`PyDynamic.deconvolution` module implements methods for the design of digital deconvolution filters
by least-squares fitting to the reciprocal of a given frequency response with associated uncertainties.
"""

# See http://mathmet.org/projects/14SIP08 and
# https://www.github.com/eichstaedtPTB/PyDynamic

from .fit_filter import LSFIR, LSIIR, LSFIR_unc, LSFIR_uncMC, LSIIR_unc, FreqResp2RealImag

__all__ = ['LSFIR',
		   'LSIIR',
		   'LSFIR_unc',
		   'LSFIR_uncMC',
		   'LSIIR_unc',
		   'FreqResp2RealImag']