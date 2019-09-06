# -*- coding: utf-8 -*-
"""
The `PyDynamic.misc` module provides various functions and methods which are
used in the examples and in some of the other implemented routines.

# See http://mathmet.org/projects/14SIP08 and
# https://www.github.com/eichstaedtPTB/PyDynamic
"""

__all__ = ['db', 'grpdelay', 'mapinside', 'isstable', 'kaiser_lowpass',
           'savitzky_golay', 'impinvar', 'sos_absphase', 'sos_realimag',
           'sos_FreqResp', 'sos_phys2filter', 'shocklikeGaussian',
           'GaussianPulse', 'squarepulse', 'rect', 'corr_noise',
           'sine', 'print_vec', 'print_mat', 'make_semiposdef',
           'FreqResp2RealImag', 'ARMA']

from .SecondOrderSystem import sos_FreqResp, sos_phys2filter, sos_absphase, \
    sos_realimag
from .filterstuff import db, grpdelay, mapinside, isstable, kaiser_lowpass, \
    savitzky_golay
from .impinvar import impinvar
from .testsignals import shocklikeGaussian, GaussianPulse, squarepulse, rect, \
    corr_noise, sine
from .noise import ARMA
from .tools import print_mat, print_vec, make_semiposdef, FreqResp2RealImag
