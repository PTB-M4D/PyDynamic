# PyDynamic
[![CircleCI](https://circleci.com/gh/PTB-PSt1/PyDynamic.svg?style=shield)](https://circleci.com/gh/PTB-PSt1/PyDynamic)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/397eebc52073457a824e5657c305dc92)](https://www.codacy.com/app/PTB-PSt1/PyDynamic?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=PTB-PSt1/PyDynamic&amp;utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/pydynamic/badge/?version=latest)](https://pydynamic.readthedocs.io/?badge=latest)
[![Codecov Badge](https://codecov.io/gh/PTB-PSt1/PyDynamic/branch/master/graph/badge.svg)](https://codecov.io/gh/PTB-PSt1/PyDynamic)
[![DOI](https://zenodo.org/badge/34848642.svg)](https://zenodo.org/badge/latestdoi/34848642)

## Python package for the analysis of dynamic measurements

The goal of this package is to provide a starting point for users in metrology and
related areas who deal with time-dependent, i.e. *dynamic*, measurements. The
initial version of this software was developed as part of a joint research project of
the national metrology institutes from Germany and the UK, i.e. [Physikalisch
-Technische Bundesanstalt](http://www.ptb.de/cms/en.html) and the [National Physical
 Laboratory](http://www.npl.co.uk).

Further development and explicit use of PyDynamic is part of the European research
project [EMPIR 17IND12 Met4FoF](http://met4fof.eu) and the German research project
[FAMOUS](https://famous-project.eu).

PyDynamic offers propagation of *uncertainties* for

- application of the discrete Fourier transform and its inverse
- filtering with an FIR or IIR filter with uncertain coefficients
- design of a FIR filter as the inverse of a frequency response with 
  uncertain coefficients
- design on an IIR filter as the inverse of a frequency response with 
  uncertain coefficients
- deconvolution in the frequency domain by division
- multiplication in the frequency domain
- transformation from amplitude and phase to a representation by real and 
  imaginary parts

For the validation of the propagation of uncertainties, the Monte-Carlo 
method can be applied using a memory-efficient implementation of Monte-Carlo 
for digital filtering.

The documentation for PyDynamic can be found on
[ReadTheDocs](http://pydynamic.readthedocs.io)

## Installation

If you just want to use the software, the easiest way is to run from your 
system's command line

```shell script
pip install PyDynamic
```

This will download the latest version from the Python package repository
and copy it into your local folder of third-party libraries. Note that
PyDynamic runs with **Python versions 3.5 to 3.8**. Usage in any Python
environment on your computer is then possible by

```python
import PyDynamic
```

or, for example, for the module containing the Fourier domain uncertainty 
methods:

```python
from PyDynamic.uncertainty import propagate_DFT
```

Updates can then be installed via

```shell script
pip install --upgrade PyDynamic
```

For collaboration we recommend forking the repository as described [here
](https://help.github.com/en/articles/fork-a-repo), apply the changes and 
open a Pull Request on GitHub as described [here
](https://help.github.com/en/articles/creating-a-pull-request). In this way 
any changes to PyDynamic can be applied very easily.

If you have downloaded this software, we would be very thankful for letting 
us know. You may, for instance, drop an email to one of the [authors
](https://github.com/PTB-PSt1/PyDynamic/graphs/contributors) (e.g. 
[Sascha Eichstädt](mailto:sascha.eichstaedt@ptb.de), [Björn Ludwig
](mailto:bjoern.ludwig@ptb.de) or [Maximilian Gruber
](mailto:maximilian.gruber@ptb.de))

### Examples
Uncertainty propagation for the application of a FIR filter with coefficients
*b* with which an uncertainty *ub* is associated. The filter input signal is
*x* with known noise standard deviation *sigma*. The filter output signal 
is *y* with associated uncertainty *uy*.

```python
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter
y, uy = FIRuncFilter(x, sigma, b, ub)    
```

Uncertainty propagation through the application of the discrete Fourier 
transform (DFT). The time domain signal is *x* with associated squared 
uncertainty *ux*. The result of the DFT is the vector *X* of real and 
imaginary parts of the DFT applied to *x* and the associated uncertainty *UX*.

```python
from PyDynamic.uncertainty.propagate_DFT import GUM_DFT
X, UX = GUM_DFT(x, ux)
```

Sequential application of the Monte Carlo method for uncertainty propagation 
for the case of filtering a time domain signal *x* with an IIR filter *b,a* 
with uncertainty associated with the filter coefficients *Uab* and signal 
noise standard deviation *sigma*. The filter output is the signal *y and the 
Monte Carlo method calculates point-wise uncertainties *uy* and coverage 
intervals *Py* corresponding to the specified percentiles.

```python
from PyDynamic.uncertainty.propagate_MonteCarlo import SMC
y, uy, Py = SMC(x, sigma, b, a, Uab, runs=1000, Perc=[0.025,0.975])
```

![PyDynamic Workflow Deconvolution](http://mathmet.org/projects/14SIP08/Deconvolution.png) 

## Roadmap

1. Implementation of robust measurement (sensor) models
2. Extension to more complex noise and uncertainty models

## Citation

If you publish results obtained with the help of PyDynamic, please use the above linked
[Zenodo DOI](https://zenodo.org/badge/latestdoi/34848642) for the code itself or cite 

Sascha Eichstädt, Clemens Elster, Ian M. Smith, and Trevor J. Esward
*Evaluation of dynamic measurement uncertainty – an open-source software 
package to bridge theory and practice*
**J. Sens. Sens. Syst.**, 6, 97-105, 2017, DOI: [10.5194/jsss-6-97-2017
](https://doi.org/10.5194/jsss-6-97-2017)

## Acknowledgement

Part of this work is developed as part of the Joint Research Project [17IND12 Met4FoF
](http://met4fof.eu) of the European Metrology Programme for Innovation and 
Research (EMPIR).

This work was part of the Joint Support for Impact project [14SIP08
](http://mathmet.org/projects/14SIP08) of the European Metrology Programme for
Innovation and Research (EMPIR). The [EMPIR](http://msu.euramet.org) is 
jointly funded by the EMPIR participating countries within EURAMET and the 
European Union.

## Disclaimer

This software is developed at Physikalisch-Technische Bundesanstalt (PTB). The
software is made available "as is" free of cost. PTB assumes no responsibility
whatsoever for its use by other parties, and makes no guarantees, expressed or
implied, about its quality, reliability, safety, suitability or any other
characteristic. In no event will PTB be liable for any direct, indirect or
consequential damage arising in connection with the use of this software.

## License

PyDynamic is distributed under the LGPLv3 license with the exception of the 
module `impinvar.py` in the package `misc`, which is distributed under the 
GPLv3 license. 
