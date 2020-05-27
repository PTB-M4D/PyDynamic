Design of deconvolution filters
===============================

.. deprecated:: 1.2.71
    The package *deconvolution* will be combined with the package *identification* and
    renamed to *model_estimation* in the next major release 2.0.0. From then on you
    should only use the new package :doc:`PyDynamic.model_estimation` instead. The
    functions :func:`LSFIR`, :func:`LSFIR_unc`, :func:`LSIIR`, :func:`LSIIR_unc`,
    :func:`LSFIR_uncMC` are then prefixed with an "inv" for "inverse", indicating the
    treatment of the reciprocal of frequency response values. Please use the new
    function names (e.g. :func:`PyDynamic.model_estimation.fit_filter.invLSIIR_unc`)
    starting from version 1.4.1. The old function names without preceding "inv" will
    only be preserved until the release prior to version 2.0.0.

The estimation of the measurand in the analysis of dynamic measurements
typically corresponds to a deconvolution problem. Therefore, a digital filter
can be designed whose input is the measured system output signal and whose
output is an estimate of the measurand. This module implements methods for
the design of such filters given an array of frequency response values with
associated uncertainties for the measurement system.

The package for now still contains the following module:

* :mod:`PyDynamic.deconvolution.fit_filter`: design of digital deconvolution filters

Digital deconvolution filters
-----------------------------

.. automodule:: PyDynamic.deconvolution.fit_filter
    :noindex:
