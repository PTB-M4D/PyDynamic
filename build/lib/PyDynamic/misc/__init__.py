# -*- coding: utf-8 -*-
"""
The `PyDynamic.misc` module provides various functions and methods which are used in the examples and in some of the
other implemented routines.
"""

from .filterstuff import db, grpdelay, mapinside, isstable, kaiser_lowpass, savitzky_golay

from .impinvar import impinvar

from .SecondOrderSystem import sos_FreqResp, sos_phys2filter, sos_absphase, sos_realimag

from .testsignals import shocklikeGaussian, GaussianPulse, squarepulse, rect, corr_noise

from .tools import print_mat, print_vec, make_semiposdef

__all__ = ['db', 'grpdelay', 'mapinside', 'isstable', 'kaiser_lowpass', 'savitzky_golay',
		   'impinvar', 'sos_absphase', 'sos_realimag', 'sos_FreqResp', 'sos_phys2filter',
		   'shocklikeGaussian','GaussianPulse','squarepulse','rect','corr_noise',
		   'print_vec','print_mat','make_semiposdef']