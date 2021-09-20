# -*- coding: utf-8 -*-
"""
.. deprecated:: 1.2.71
    The package *identification* will be combined with the package *deconvolution* and
    renamed to *model_estimation* in the next major release 2.0.0. From version 1.4.1 on
    you should only use the new package :doc:`PyDynamic.model_estimation` instead.

The module :mod:`PyDynamic.identification.fit_transfer` contains several functions
for the identification of transfer function models.

This module for now still contains the following function:

* :func:`fit_sos`: Fit second-order model to complex-valued frequency response

"""
import warnings

from ..model_estimation.fit_transfer import fit_som

__all__ = ["fit_sos"]

warnings.simplefilter("default")
warnings.warn(
    "The package *identification* will be combined with the package "
    "*deconvolution* and renamed to *model_estimation* in the "
    "next major release 2.0.0. From version 1.4.1 on you should only use "
    "the new package *model_estimation* instead.",
    DeprecationWarning,
)


def fit_sos(f, H, UH=None, weighting=None, MCruns=None, scaling=1e-3):
    """.. deprecated:: 1.4.1 Please use :func:`PyDynamic.model_estimation.fit_som`"""
    return fit_som(f=f, H=H, UH=UH, weighting=weighting, MCruns=MCruns, scaling=scaling)
