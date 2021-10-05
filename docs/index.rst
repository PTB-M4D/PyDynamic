PyDynamic - Analysis of dynamic measurements
============================================

PyDynamic is a Python software package developed jointly by mathematicians from
`Physikalisch-Technische Bundesanstalt <www.ptb.de>`_ (Germany) and
`National Physical Laboratory <www.npl.co.uk>`_ (UK) as part of the joint
European Research Project
`EMPIR 14SIP08 Dynamic <https://www.euramet.org/research-innovation/search
-research-projects/details/project/standards-and-software-to-maximise-end-user
-uptake-of-nmi-calibrations-of-dynamic-force-torque-and/>`_.

For the PyDynamic homepage go to
`GitHub <https://github.com/PTB-M4D/PyDynamic>`_.

*PyDynamic* is written in Python 3 and strives to run with `all Python versions with
upstream support <https://devguide.python.org/#status-of-python-branches>`_. Currently
it is tested to work with Python 3.6 to 3.9.


.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   Getting started
   Examples
   CONTRIBUTING

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   PyDynamic.uncertainty
   PyDynamic.model_estimation
   PyDynamic.deconvolution
   PyDynamic.identification
   PyDynamic.misc

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   Tutorials

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
==========

.. [Eichst2016] S. Eichstädt und V. Wilkens
                GUM2DFT — a software tool for uncertainty evaluation of
                transient signals in the frequency domain.
                Meas. Sci. Technol., 27(5), 055001, 2016.
                https://dx.doi.org/10.1088/0957-0233/27/5/055001

.. [Eichst2012] S. Eichstädt, A. Link, P. M. Harris and C. Elster
                Efficient implementation of a Monte Carlo method for
                uncertainty evaluation in dynamic measurements
                Metrologia, vol 49(3), 401
                https://dx.doi.org/10.1088/0026-1394/49/3/401

.. [Eichst2010] S. Eichstädt, C. Elster, T. J. Esward and J. P. Hessling
                Deconvolution filters for the analysis of dynamic measurement
                processes: a tutorial
                Metrologia, vol. 47, nr. 5
                https://stacks.iop.org/0026-1394/47/i=5/a=003?key=crossref.310be1c501bb6b6c2056bc9d22ec93d4

.. [Elster2008] C. Elster and A. Link
                Uncertainty evaluation for dynamic measurements modelled by a
                linear time-invariant system
                Metrologia, vol 45 464-473, 2008
                https://dx.doi.org/10.1088/0026-1394/45/4/013

.. [Link2009]   A. Link and C. Elster
                Uncertainty evaluation for IIR filtering using a state-space
                approach
                Meas. Sci. Technol. vol. 20, 2009
                https://dx.doi.org/10.1088/0957-0233/20/5/055104

.. [Vuer1996]   R. Vuerinckx, Y. Rolain, J. Schoukens and R. Pintelon
                Design of stable IIR filters in the complex domain by
                automatic delay selection
                IEEE Trans. Signal Proc., 44, 2339-44, 1996
                https://dx.doi.org/10.1109/78.536690

.. [Smith]      Smith, J.O. Introduction to Digital Filters with Audio
                Applications, https://ccrma.stanford.edu/~jos/filters/, online
                book

.. [Savitzky]   A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
                Data by Simplified Least Squares Procedures. Analytical
                Chemistry, 1964, 36 (8), pp 1627-1639.

.. [NumRec]     Numerical Recipes 3rd Edition: The Art of Scientific Computing
                W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
                Cambridge University Press ISBN-13: 9780521880688

.. [White2017]  White, D.R. Int J Thermophys (2017) 38: 39.
                https://doi.org/10.1007/s10765-016-2174-6
