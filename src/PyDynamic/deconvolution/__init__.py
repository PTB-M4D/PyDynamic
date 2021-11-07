"""
.. deprecated:: 2.0.0
    The module *deconvolution* is combined with the module *identification* and
    renamed to :mod:`PyDynamic.model_estimation` since the last major release 2.0.0.
    *deconvolution* might be removed any time. Please switch to the current module
    immediately. The previously known functions `LSFIR()`, `LSFIR_unc()`, `LSIIR()`,
    `LSIIR_unc()`, `LSFIR_uncMC()` were transferred and merged into
    :func:`PyDynamic.model_estimation.fit_filter.LSIIR` and
    :func:`PyDynamic.model_estimation.fit_filter.LSFIR`. Set `inv=true` to fit
    against the reciprocal of the frequency response like with the previous versions
    in the module *deconvolution*.
"""

from .fit_filter import LSFIR, LSFIR_unc, LSFIR_uncMC, LSIIR, LSIIR_unc
