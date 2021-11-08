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

warning = (
    "The module *deconvolution* is combined with the module *identification* and "
    "renamed to 'PyDynamic.model_estimation' since the last major release 2.0.0. "
    "'PyDynamic.deconvolution' might be removed any time. Please switch to the "
    "current module immediately. The previously known functions 'LSFIR()', "
    "'LSFIR_unc()', 'LSIIR()', 'LSIIR_unc()', 'LSFIR_uncMC()' were transferred and "
    "merged into 'PyDynamic.model_estimation.fit_filter.LSIIR' and "
    "'PyDynamic.model_estimation.fit_filter.LSFIR'. Set 'inv = true' to fit "
    "against the reciprocal of the frequency response like with the previous versions "
    "in the module *deconvolution*. Please use the current function name "
)


def LSFIR(*args, **kwargs):
    """.. deprecated:: 2.0.0 Please use :func:`PyDynamic.model_estimation.LSFIR`"""
    raise DeprecationWarning(f"{warning}'PyDynamic.model_estimation.LSFIR'.")


def LSFIR_unc(*args, **kwargs):
    """.. deprecated:: 2.0.0 Please use :func:`PyDynamic.model_estimation.LSFIR`"""
    raise DeprecationWarning(f"{warning}'PyDynamic.model_estimation.LSFIR'.")


def LSFIR_uncMC(*args, **kwargs):
    """
    .. deprecated:: 2.0.0
        Please use :func:`PyDynamic.model_estimation.LSFIR`
    """
    raise DeprecationWarning(f"{warning}:func:`PyDynamic.model_estimation.LSFIR`.")


def LSIIR(*args, **kwargs):
    """.. deprecated:: 2.0.0 Please use :func:`PyDynamic.model_estimation.LSIIR`"""
    raise DeprecationWarning(f"{warning}'PyDynamic.model_estimation.LSIIR'.")


def LSIIR_unc(*args, **kwargs):
    """.. deprecated:: 2.0.0 Please use :func:`PyDynamic.model_estimation.LSIIR`"""
    raise DeprecationWarning(f"{warning}'PyDynamic.model_estimation.LSIIR'.")
