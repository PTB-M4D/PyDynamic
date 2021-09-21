# -*- coding: utf-8 -*-
"""
.. deprecated:: 2.0.0
    The module :mod:`PyDynamic.uncertainty.interpolation` is renamed to
    :mod:`PyDynamic.uncertainty.interpolate` since the last major release 2.0.0. It
    might be removed any time. Please switch to the current module immediately.
"""

__all__ = ["interp1d_unc"]


def interp1d_unc(*args, **kwargs):
    """
    .. deprecated:: 2.0.0
    Please use :func:`PyDynamic.uncertainty.interpolate.interp1d_unc`
    """
    raise DeprecationWarning(
        "The module *interpolation* is renamed to "
        ":mod:`PyDynamic.uncertainty.interpolate` since the last major release 2.0.0. "
        "Please switch to the current module immediately and use the current function "
        ":func:`PyDynamic.uncertainty.interpolate.interp1d_unc`.",
    )
