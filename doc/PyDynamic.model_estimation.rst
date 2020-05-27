Model estimation
================

The estimation of the measurand in the analysis of dynamic measurements typically
corresponds to a deconvolution problem. Therefore, a digital filter can be designed
whose input is the measured system output signal and whose output is an estimate of the
measurand. The package :doc:`PyDynamic.model_estimation` implements methods for the
design of such filters given an array of frequency response values or the reciprocal of
frequency response values with associated uncertainties for the measurement system.

The package :doc:`PyDynamic.model_estimation` also contains a function for the
identification of transfer function models.

The package consists of the following modules:

* :mod:`PyDynamic.model_estimation.fit_filter`: least-squares fit to a given complex
  frequency response or its reciprocal
* :mod:`PyDynamic.model_estimation.fit_transfer`: identification of transfer function
  models

Fitting filters to frequency response or reciprocal
---------------------------------------------------

.. automodule:: PyDynamic.model_estimation.fit_filter
    :members:

Identification of transfer function models
------------------------------------------

.. automodule:: PyDynamic.model_estimation.fit_transfer
    :members:
