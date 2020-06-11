Fitting filters and transfer functions models
=============================================

.. deprecated:: 1.2.71
    The package *identification* will be combined with the package *deconvolution* and
    renamed to *model_estimation* in the next major release 2.0.0. From version 1.4.1 on
    you should only use the new package :doc:`PyDynamic.model_estimation` instead.

The package for now still contains the following modules:

* :mod:`PyDynamic.identification.fit_filter`: least-squares fit to a given complex
  frequency response
* :mod:`PyDynamic.identification.fit_transfer`: identification of transfer function
  models

Fitting filters to frequency response
-------------------------------------

.. automodule:: PyDynamic.identification.fit_filter
    :noindex:

Identification of transfer function models
------------------------------------------

.. automodule:: PyDynamic.identification.fit_transfer
    :noindex:
