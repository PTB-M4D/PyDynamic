Evaluation of uncertainties
===========================

The evaluation of uncertainties is a fundamental part of the measurement
analysis in metrology. The analysis of dynamic measurements typically
involves methods from signal processing, such as digital filtering or
application of the discrete Fourier transform (DFT). For most tasks, methods
are readily available, for instance, as part of :mod:`scipy.signals`. This
module of PyDynamic provides the corresponding methods for the evaluation of
uncertainties.

Uncertainty evaluation for the DFT
----------------------------------

.. automodule:: PyDynamic.uncertainty.propagate_DFT
    :members:

Uncertainty evaluation for digital filtering
--------------------------------------------

.. automodule:: PyDynamic.uncertainty.propagate_filter
    :members:

Monte Carlo methods for digital filtering
-----------------------------------------

.. automodule:: PyDynamic.uncertainty.propagate_MonteCarlo
    :members:
