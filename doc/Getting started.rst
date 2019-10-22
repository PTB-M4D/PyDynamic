Getting started
===============

Installation
------------
If you just want to use the software, the easiest way is to run from your
system's command line

.. code-block:: bash

    pip install PyDynamic

This will download the latest version from the Python package repository and
copy it into your local folder of third-party libraries. Usage in any Python
environment on your computer is then possible by

.. code-block:: python

    import PyDynamic

or, for example, for the module containing the Fourier domain uncertainty
methods:

.. code-block:: python

    from PyDynamic.uncertainty import propagate_DFT

Updates of the software can be installed via

.. code-block:: bash

    pip install --upgrade PyDynamic

For collaboration we recommend using `Github Desktop <https://desktop.github
.com>`_ or any other git-compatible version control software and cloning the
`repository <https://github.com/PTB-PSt1/PyDynamic>`_. In this way, any
updates to the software will be highlighted in the version control software
and can be applied very easily.

If you have downloaded this software, we would be very thankful for letting
us know. You may, for instance, drop an email to one of the `authors
<https://github.com/PTB-PSt1/PyDynamic/graphs/contributors>`_.

Quick Examples
--------------
On the project website you can find various examples illustrating the
application of the software in the examples folder. Here is just a short list
to get you started.

Uncertainty propagation for the application of an FIR filter with coefficients
*b* with which an uncertainty *ub* is associated. The filter input signal is
*x* with known noise standard deviation *sigma*. The filter output signal
is *y* with associated uncertainty *uy*.

.. code-block:: python

    from PyDynamic.uncertainty.propagate_filter import FIRuncFilter
    y, uy = FIRuncFilter(x, sigma, b, ub)

Uncertainty propagation through the application of the discrete Fourier
transform (DFT). The time domain signal is *x* with associated squared
uncertainty *ux*. The result of the DFT is the vector *X* of real and
imaginary parts of the DFT applied to *x* and the associated uncertainty *UX*.

.. code-block:: python

    from PyDynamic.uncertainty.propagate_DFT import GUM_DFT
    X, UX = GUM_DFT(x, ux)

Sequential application of the Monte Carlo method for uncertainty propagation
for the case of filtering a time domain signal *x* with an IIR filter *b,a*
with uncertainty associated with the filter coefficients *Uab* and signal
noise standard deviation *sigma*. The filter output is the signal *y* and the
Monte Carlo method calculates point-wise uncertainties *uy* and coverage
intervals *Py* corresponding to the specified percentiles.

.. code-block:: python

  from PyDynamic.uncertainty.propagate_MonteCarlo import SMC
  y, uy, Py = SMC(x, sigma, b, a, Uab, runs=1000, Perc=[0.025,0.975])

Detailed examples
-----------------

.. toctree::
   :maxdepth: 1

   Deconvolution by FIR.rst
   Uncertainty propagation for IIR filters.rst
   Deconvolution in the DFT domain.rst
