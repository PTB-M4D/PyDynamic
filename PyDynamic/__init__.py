"""
import misc
import identification
import deconvolution
import uncertainty

# See http://mathmet.org/projects/14SIP08 and
# https://www.github.com/eichstaedtPTB/PyDynamic
"""

from .deconvolution.fit_filter import LSFIR_unc, LSFIR_uncMC, LSIIR_unc
from .identification import *
from .misc import *
from .model_estimation import *
from .uncertainty import *

__version__ = "1.3.1"

__all__ = [
    "iLSFIR",
    "iLSIIR",
    "iLSFIR_unc",
    "iLSFIR_uncMC",
    "iLSIIR_unc",
    "LSFIR",
    "LSIIR",
    "LSFIR_unc",
    "LSFIR_uncMC",
    "LSIIR_unc",
    "fit_sos",
    "FreqResp2RealImag",
    "GUM_DFT",
    "GUM_iDFT",
    "DFT_deconv",
    "DFT_multiply",
    "DFT2AmpPhase",
    "AmpPhase2DFT",
    "AmpPhase2Time",
    "Time2AmpPhase",
    "FIRuncFilter",
    "IIRuncFilter",
    "MC",
    "SMC",
    "UMC",
    "UMC_generic",
    "interp1d_unc",
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
    "print_vec",
    "print_mat",
    "make_semiposdef",
    "make_equidistant",
]
