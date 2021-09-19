# -*- coding: utf-8 -*-
"""
.. deprecated:: 2.0.0
    The module :mod:`PyDynamic.uncertainty.interpolation` will be renamed to
    :mod:`PyDynamic.uncertainty.interpolate` in the next major release 2.0.0. From
    version 1.4.3 on you should only use the new module instead.

The :mod:`PyDynamic.uncertainty.interpolation` module implements methods for
the propagation of uncertainties in the application of standard interpolation methods
as provided by :class:`scipy.interpolate.interp1d`.

This module for now still contains the following function:

* :func:`interp1d_unc`: Interpolate arbitrary time series considering the associated
  uncertainties
"""
import warnings

from .interpolate import interp1d_unc

__all__ = ["interp1d_unc"]

warnings.simplefilter("default")
warnings.warn(
    "The module :mod:`PyDynamic.uncertainty.interpolation` will be renamed to "
    ":mod:`PyDynamic.uncertainty.interpolate` in the next major release 2.0.0. From "
    "version 1.4.3 on you should only use the new module instead.",
    PendingDeprecationWarning,
)
