# -*- coding: utf-8 -*-
"""
.. deprecated:: 1.2.71
    The package *identification* will be combined with the package *deconvolution* and
    renamed to *model_estimation* in the next major release 2.0.0. From version 1.4.1 on
    you should only use the new package :doc:`PyDynamic.model_estimation` instead.

This module contains several functions to carry out a least-squares fit to a
given complex frequency response.

This module for now still contains the following functions:

* :func:`LSIIR`: Least-squares IIR filter fit to a given frequency response
* :func:`LSFIR`: Least-squares fit of a digital FIR filter to a given frequency
  response
"""
import warnings

from ..model_estimation.fit_filter import LSFIR, LSIIR

__all__ = ["LSIIR", "LSFIR"]

warnings.simplefilter("default")
warnings.warn(
    "The module *identification* will be combined with the module "
    "*deconvolution* and renamed to *model_estimation* in the "
    "next major release 2.0.0. From version 1.4.1 on you should only use "
    "the new module *model_estimation* instead.",
    DeprecationWarning,
)
