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
- 1-dimensional interpolation

For the validation of the propagation of uncertainties, the Monte-Carlo
method can be applied using a memory-efficient implementation of Monte-Carlo
for digital filtering.

The documentation for PyDynamic can be found on
[ReadTheDocs](http://pydynamic.readthedocs.io)

## Package diagram

The fundamental structure of PyDynamic is shown in the following figure.

![PyDynamic package diagram](https://raw.githubusercontent.com/PTB-PSt1/PyDynamic/master/docs/PyDynamic_package_diagram.png)

However, imports should generally be possible without explicitly naming all packages
and modules in the path, so that for example the following import statements are all
equivalent.

```python
from PyDynamic.uncertainty.interpolate import make_equidistant
from PyDynamic.uncertainty import make_equidistant
from PyDynamic import make_equidistant
```

## Installation

There is a [quick way](#quick-setup-not-recommended) to get started but we advise to
setup a virtual environment and guide through the process in the section
[Proper Python setup with virtual environment](#proper-python-setup-with-virtual-environment-recommended)

### Quick setup (**not recommended**)

If you just want to use the software, the easiest way is to run from your
system's command line

```shell
pip install --user PyDynamic
```

This will download the latest version from the Python package repository
and copy it into your local folder of third-party libraries. Note that
PyDynamic runs with **Python versions 3.6 to 3.8**. Usage in any Python
environment on your computer is then possible by

```python
import PyDynamic
```

or, for example, for the module containing the Fourier domain uncertainty
methods:

```python
from PyDynamic.uncertainty import propagate_DFT
```

#### Updating to the newest version

Updates can then be installed via

```shell
pip install --user --upgrade PyDynamic
```

### Proper Python setup with virtual environment (**recommended**)

The setup described above allows the quick and easy use of PyDynamic, but it also has
its downsides. When working with Python we should rather always work in so-called
virtual environments, in which our project specific dependencies are satisfied
without polluting or breaking other projects' dependencies and to avoid breaking all
our dependencies in case of an update of our Python distribution.

If you are not familiar with [Python virtual environments
](https://docs.python.org/3/glossary.html#term-virtual-environment) you can get the
motivation and an insight into the mechanism in the
[official docs](https://docs.python.org/3/tutorial/venv.html).

#### Create a virtual environment and install requirements

Creating a virtual environment with Python built-in tools is easy and explained
in more detail in the
[official docs of Python itself](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments).

It boils down to creating an environment anywhere on your computer, then activate
it and finally install PyDynamic and its dependencies.

##### _venv_ creation and installation in Windows

In your Windows command prompt execute the following:

```shell
> py -3 -m venv LOCAL\PATH\TO\ENVS\PyDynamic_venv
> LOCAL\PATH\TO\ENVS\PyDynamic_venv\Scripts\activate.bat
(PyDynamic_venv) > pip install PyDynamic
```

##### _venv_ creation and installation on Mac and Linux

In your terminal execute the following:

```shell
$ python3 -m venv /LOCAL/PATH/TO/ENVS/PyDynamic_venv
$  /LOCAL/PATH/TO/ENVS/PyDynamic_venv/bin/activate
(PyDynamic_venv) $ pip install PyDynamic
```

#### Updating to the newest version

Updates can then be installed on all platforms after activating the virtual environment
via:

```shell
(PyDynamic_venv) $ pip install --upgrade PyDynamic
```

### Optional Jupyter Notebook dependencies

If you are familiar with Jupyter Notebooks, you find some examples in the _examples_ and
the _tutorials_ subfolders of the source code repository. To execute these you need
additional dependencies which you get by appending `[examples]` to PyDynamic in all
of the above, e.g.

```shell
(PyDynamic_venv) $ pip install PyDynamic[examples]
```

### Install known to work dependencies' versions

In case errors arise within PyDynamic, the first thing you can try is installing the
known to work configuration of dependencies against which we run our test suite. This
you can easily achieve with our version specific requirements files. First you need
to install our dependency management package _pip-tools_, then find the Python
version you are using with PyDynamic and finally install the provided dependency
versions for your specific Python version. This is all done with the following
sequence of commands after activating. Change the suffix `-py38` according to the
 Python version you find after executing `(PyDynamic_venv) $ python --version`:

```shell
(PyDynamic_venv) $ pip install --upgrade pip-tools
Collecting pip-tools
[...]
Successfully installed pip-tools-5.2.1
(PyDynamic_venv) $ python --version
Python 3.8.3
(PyDynamic_venv) $ pip-sync requirements/dev-requirements-py38.txt requirements/requirements-py38.txt
Collecting [...]
[...]
Successfully installed [...]
(PyDynamic_venv) $
```

## Contributing to PyDynamic

If you want to contribute code to the project you find additional set up and related
information in our [Contribution advices and tips](docs/CONTRIBUTING.md).

If you have a feature request please take a look at the roadmap and the links
provided there to find out more about planned and ongoing developments. If you
have the feeling, something is missing, let us know by opening an issue.

If you have downloaded this software, we would be very thankful for letting
us know. You may, for instance, drop an email to one of the [authors
](https://github.com/PTB-PSt1/PyDynamic/graphs/contributors) (e.g.
[Sascha Eichstädt](mailto:sascha.eichstaedt@ptb.de), [Björn Ludwig
](mailto:bjoern.ludwig@ptb.de) or [Maximilian Gruber
](mailto:maximilian.gruber@ptb.de))

## Examples

We have collected extended material for an easier introduction to PyDynamic in the two
subfolders _examples_ and _tutorials_. In various Jupyter Notebooks and scripts we
demonstrate the use of the provided methods to aid the first steps in PyDynamic. New
features are introduced with an example from the beginning if feasible. We are currently
moving this supporting collection to an external repository on GitHub. They will be
available at
[github.com/PTB-PSt1/PyDynamic_tutorials](https://github.com/PTB-PSt1/PyDynamic_tutorials)
in the near future.

Uncertainty propagation for the application of a FIR filter with coefficients *b* with
which an uncertainty *ub* is associated. The filter input signal is *x* with known noise
standard deviation *sigma*. The filter output signal is *y* with associated uncertainty
*uy*.

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
for the case of filtering a time domain signal _x_ with an IIR filter _b_, _a_
with uncertainty associated with the filter coefficients _Uab_ and signal
noise standard deviation _sigma_. The filter output is the signal _y_ and the
Monte Carlo method calculates point-wise uncertainties _uy_ and coverage
intervals _Py_ corresponding to the specified percentiles.

```python
from PyDynamic.uncertainty.propagate_MonteCarlo import SMC
y, uy, Py = SMC(x, sigma, b, a, Uab, runs=1000, Perc=[0.025,0.975])
```

![PyDynamic Workflow Deconvolution](https://mathmet.org/projects/14SIP08/Deconvolution.png)

## Roadmap

1. Implementation of robust measurement (sensor) models
1. Extension to more complex noise and uncertainty models
1. Introducing uncertainty propagation for Kalman filters

For a comprehensive overview of current development activities and upcoming tasks,
take a look at the [project board](https://github.com/PTB-PSt1/PyDynamic/projects/1),
[issues](https://github.com/PTB-PSt1/PyDynamic/issues) and
[pull requests](https://github.com/PTB-PSt1/PyDynamic/pulls).

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
