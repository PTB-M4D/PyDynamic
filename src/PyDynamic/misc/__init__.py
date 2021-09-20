# -*- coding: utf-8 -*-
"""
The `PyDynamic.misc` module provides various functions and methods which are
used in the examples and in some of the other implemented routines.

.. seealso::

   - `initial project website <https://www.euramet.org/research-innovation/search
     -research-projects/details/project/standards-and-software-to-maximise-end-user
     -uptake-of-nmi-calibrations-of-dynamic-force-torque-and/>`_
   - `GitHub website <https://www.github.com/PTB-M4D/PyDynamic>`_
"""

__all__ = [
    "db",
    "grpdelay",
    "mapinside",
    "isstable",
    "kaiser_lowpass",
    "savitzky_golay",
    "impinvar",
    "sos_absphase",
    "sos_realimag",
    "sos_FreqResp",
    "sos_phys2filter",
    "shocklikeGaussian",
    "GaussianPulse",
    "squarepulse",
    "rect",
    "corr_noise",
    "sine",
    "ARMA",
    "print_mat",
    "print_vec",
    "make_semiposdef",
    "FreqResp2RealImag",
    "make_equidistant",
    "trimOrPad",
    "progress_bar",
    "shift_uncertainty",
    "is_vector",
    "is_2d_matrix",
    "number_of_rows_equals_vector_dim",
]

from .SecondOrderSystem import sos_FreqResp, sos_phys2filter, sos_absphase, sos_realimag
from .filterstuff import (
    db,
    grpdelay,
    mapinside,
    isstable,
    kaiser_lowpass,
    savitzky_golay,
)
from .impinvar import impinvar
from .testsignals import (
    shocklikeGaussian,
    GaussianPulse,
    squarepulse,
    rect,
    corr_noise,
    sine,
)
from .noise import ARMA
from .tools import (
    print_mat,
    print_vec,
    make_semiposdef,
    FreqResp2RealImag,
    make_equidistant,
    trimOrPad,
    progress_bar,
    shift_uncertainty,
    is_vector,
    is_2d_matrix,
    number_of_rows_equals_vector_dim,
)
