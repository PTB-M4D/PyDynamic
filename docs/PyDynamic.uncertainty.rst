Evaluation of uncertainties
===========================

The evaluation of uncertainties is a fundamental part of the measurement
analysis in metrology. The analysis of dynamic measurements typically
involves methods from signal processing, such as digital filtering, the discrete
Fourier transform (DFT), or simple tasks like interpolation. For most of these tasks,
methods are readily available, for instance, as part of :mod:`scipy.signal`. This
module of PyDynamic provides the corresponding methods for the evaluation of
uncertainties.

The package consists of the following modules:

* :mod:`PyDynamic.uncertainty.propagate_DFT`: Uncertainty evaluation for the DFT
* :mod:`PyDynamic.uncertainty.propagate_DWT`: Uncertainty evaluation for the DWT
* :mod:`PyDynamic.uncertainty.propagate_convolution`: Uncertainty evaluation for
  convolutions
* :mod:`PyDynamic.uncertainty.propagate_filter`: Uncertainty evaluation for digital
  filtering
* :mod:`PyDynamic.uncertainty.propagate_MonteCarlo`: Monte Carlo methods for digital
  filtering
* :mod:`PyDynamic.uncertainty.interpolate`: Uncertainty evaluation for interpolation
* :mod:`PyDynamic.uncertainty.propagate_multiplication`: Uncertainty evaluation for 
  element-wise multiplication

Uncertainty evaluation for convolutions
---------------------------------------

.. automodule:: PyDynamic.uncertainty.propagate_convolution
    :members:

Uncertainty evaluation for the DFT
----------------------------------

.. automodule:: PyDynamic.uncertainty.propagate_DFT
    :members:

Uncertainty evaluation for the DWT
----------------------------------

.. automodule:: PyDynamic.uncertainty.propagate_DWT
    :members:

Uncertainty evaluation for digital filtering
--------------------------------------------

.. automodule:: PyDynamic.uncertainty.propagate_filter
    :members:

Monte Carlo methods for digital filtering
-----------------------------------------

.. automodule:: PyDynamic.uncertainty.propagate_MonteCarlo
    :members:

Uncertainty evaluation for interpolation
----------------------------------------

.. automodule:: PyDynamic.uncertainty.interpolate
    :members:

Uncertainty evaluation for multiplication
-----------------------------------------

.. automodule:: PyDynamic.uncertainty.propagate_multiplication
    :members:
