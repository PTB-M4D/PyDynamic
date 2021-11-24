"""Various functions to reduce redundancy in PyDynamic's codebase

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
    "multi_sine",
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
    "real_imag_2_complex",
    "separate_real_imag_of_mc_samples",
    "separate_real_imag_of_vector",
    "complex_2_real_imag",
]

from .filterstuff import (
    db,
    grpdelay,
    isstable,
    kaiser_lowpass,
    mapinside,
    savitzky_golay,
)
from .impinvar import impinvar
from .noise import ARMA
from .SecondOrderSystem import sos_absphase, sos_FreqResp, sos_phys2filter, sos_realimag
from .testsignals import (
    corr_noise,
    GaussianPulse,
    multi_sine,
    rect,
    shocklikeGaussian,
    sine,
    squarepulse,
)
from .tools import (
    complex_2_real_imag,
    FreqResp2RealImag,
    is_2d_matrix,
    is_vector,
    make_equidistant,
    make_semiposdef,
    number_of_rows_equals_vector_dim,
    print_mat,
    print_vec,
    progress_bar,
    real_imag_2_complex,
    separate_real_imag_of_mc_samples,
    separate_real_imag_of_vector,
    shift_uncertainty,
    trimOrPad,
)
