Design of deconvolution filters
===============================

The estimation of the measurand in the analysis of dynamic measurements
typically corresponds to a deconvolution problem. Therefore, a digital filter
can be designed whose input is the measured system output signal and whose
output is an estimate of the measurand. This module implements methods for
the design of such filters given an array of frequency response values with
associated uncertainties for the measurement system.

The package contains the following module:

* :mod:`PyDynamic.deconvolution.fit_filter`: design of digital deconvolution filters

Digital deconvolution filters
-----------------------------

.. automodule:: PyDynamic.deconvolution.fit_filter
    :noindex:
