"""
.. deprecated:: 2.0.0
    The module *identification* is combined with the module *deconvolution* and
    renamed to :mod:`PyDynamic.model_estimation` since the last major release 2.0.0.
    *identification* might be removed any time. Please switch to the current module
    immediately.
"""
__all__ = ["LSIIR", "LSFIR"]

warning = (
    "The module *identification* is combined with the module *deconvolution* and "
    "renamed to 'PyDynamic.model_estimation' since the last major release 2.0.0. "
    "*identification* might be removed any time. Please switch to the current module "
    "immediately. Please use the current function name "
)


def LSIIR(*args, **kwargs):
    """.. deprecated:: 2.0.0 Please use :func:`PyDynamic.model_estimation.LSIIR`"""
    raise DeprecationWarning(f"{warning}'PyDynamic.model_estimation.LSIIR'.")


def LSFIR(*args, **kwargs):
    """.. deprecated:: 2.0.0 Please use :func:`PyDynamic.model_estimation.LSFIR`"""
    raise DeprecationWarning(f"{warning}'PyDynamic.model_estimation.LSFIR'.")
