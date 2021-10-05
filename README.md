<p align="center">
  <!-- CircleCI Tests -->
  <a href="https://circleci.com/gh/PTB-M4D/PyDynamic"><img alt="CircleCI pipeline status badge" src="https://circleci.com/gh/PTB-M4D/PyDynamic.svg?style=shield"></a>
  <!-- ReadTheDocs Documentation -->
  <a href="https://pydynamic.readthedocs.io/en/latest/index.html">
    <img src="https://readthedocs.org/projects/pydynamic/badge/?version=latest">
  </a>
  <!-- CodeCov(erage) -->
  <a href="https://codecov.io/gh/PTB-M4D/PyDynamic">
    <img src="https://codecov.io/gh/PTB-M4D/PyDynamic/branch/master/graph/badge.svg"/>
  </a>
  <!-- Codacy -->
  <a href="https://www.codacy.com/gh/PTB-M4D/PyDynamic/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=PTB-M4D/PyDynamic&amp;utm_campaign=Badge_Grade">
    <img src="https://app.codacy.com/project/badge/Grade/db86b58d6fa5446e8408644c8196f5e2"/>
  </a>
  <!-- PyPI Version -->
  <a href="https://pypi.org/project/pydynamic">
    <img src="https://img.shields.io/pypi/v/pydynamic.svg?label=release&color=blue&style=flat-square" alt="pypi">
  </a>
  <!-- PyPI License -->
  <a href="https://www.gnu.org/licenses/lgpl-3.0.en.html">
    <img alt="PyPI - license badge" src="https://img.shields.io/pypi/l/pydynamic?color=bright">
  </a>
  <!-- Zenodo DOI -->
  <a href="https://doi.org/10.5281/zenodo.4748367">
    <img src="https://zenodo.org/badge/34848642.svg" alt="DOI"></a>
</p>

<h1 align="center">
PyDynamic – Python package for the analysis of dynamic measurements
</h1>

<p align="justify">
The goal of this package is to provide a starting point for users in metrology and
related areas who deal with time-dependent i.e., <i>dynamic</i>, measurements. The
initial version of this software was developed as part of a joint research project of
the national metrology institutes from Germany and the UK, i.e.
<a href="https://www.ptb.de/cms/en.html">Physikalisch-Technische Bundesanstalt</a> 
and the <a href="https://www.npl.co.uk">National Physical Laboratory</a>.
</p>

<p align="justify">
Further development and explicit use of PyDynamic is part of the European research
project <a href="https://www.ptb.de/empir2018/met4fof/home/">EMPIR 17IND12 
Met4FoF</a> and the German research project <a href="https://famous-project.
eu">FAMOUS</a>.
</p>

## Features

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

![PyDynamic package diagram](https://raw.githubusercontent.com/PTB-M4D/PyDynamic/master/docs/PyDynamic_package_diagram.png)

However, imports should generally be possible without explicitly naming all packages
and modules in the path, so that for example the following import statements are all
equivalent.

```python
from PyDynamic.uncertainty.interpolate import make_equidistant
from PyDynamic.uncertainty import make_equidistant
from PyDynamic import make_equidistant
```

## Installation

There is a [quick way](#quick-setup-not-recommended) to get started, but we advise
setting up a virtual environment and guide through the process in the section
[Proper Python setup with virtual environment](#proper-python-setup-with-virtual-environment-recommended)

### Quick setup (**not recommended**)

If you just want to use the software as quick as possible, run from your system's 
command line

```shell
pip install --user PyDynamic
```

This will download the latest version from the Python package repository
and copy it into your local folder of third-party libraries. Note that
PyDynamic runs with **Python versions 3.6 to 3.9**. Usage in any Python
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
its downsides. When working with Python, we should rather always work in so-called
virtual environments. These allow to satisfy project specific dependencies without 
polluting or breaking other projects' dependencies.

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

If you are familiar with Jupyter Notebooks, you find some examples in the _examples_
subfolder of the source code repository. To execute these you need additional 
dependencies which you get by appending `[examples]` to PyDynamic in all 
the above, e.g.

```shell
(PyDynamic_venv) $ pip install PyDynamic[examples]
```

### Install known to work dependencies' versions

In case errors arise within PyDynamic, the first thing you can try is installing the
known to work configuration of dependencies against which we run our test suite. This
you can easily achieve with our version specific requirements files. First you need
to install our dependency management package _pip-tools_, then find the Python
version you are using with PyDynamic. Finally, you install the provided dependency
versions for your specific Python version. This is all done with the following
sequence of commands after activating. Change the suffix `-py38` according to the
Python version you find after executing `(PyDynamic_venv) $ python --version`:

```shell
(PyDynamic_venv) $ pip install --upgrade pip-tools
Collecting pip-tools
[...]
Successfully installed pip-tools-5.2.1
(PyDynamic_venv) $ python --version
Python 3.8.8
(PyDynamic_venv) $ pip-sync requirements/dev-requirements-py38.txt requirements/requirements-py38.txt
Collecting [...]
[...]
Successfully installed [...]
(PyDynamic_venv) $
```

## Contributing to PyDynamic

If you want to contribute code to the project you find additional set up and related
information in our [Contribution advices and tips](docs/CONTRIBUTING.md).

If you have a feature request please take a look at the roadmap, and the links
provided there to find out more about planned and ongoing developments. If you
have the feeling, something is missing, let us know by opening an issue.

If you have downloaded this software, we would be very thankful for letting
us know. You may, for instance, drop an email to one of the [authors
](https://github.com/PTB-M4D/PyDynamic/graphs/contributors) (e.g.
[Sascha Eichstädt](mailto:sascha.eichstaedt@ptb.de), [Björn Ludwig
](mailto:bjoern.ludwig@ptb.de) or [Maximilian Gruber
](mailto:maximilian.gruber@ptb.de))

## Examples

We have collected extended material for an easier introduction to PyDynamic in the
package _examples_. Detailed assistance on getting started you can find in the
corresponding sections of the docs:

* [examples](https://pydynamic.readthedocs.io/en/latest/Examples.html)
* [tutorials](https://pydynamic.readthedocs.io/en/latest/Tutorials.html)

In various Jupyter Notebooks and scripts we demonstrate the use of
the provided methods to aid the first steps in PyDynamic. New features are introduced
with an example from the beginning if feasible. We are currently moving this supporting
collection to an external repository on GitHub. They will be available at
[github.com/PTB-M4D/PyDynamic_tutorials](https://github.com/PTB-M4D/PyDynamic_tutorials) 
in the near future.

## Roadmap

1. Implementation of robust measurement (sensor) models
1. Extension to more complex noise and uncertainty models
1. Introducing uncertainty propagation for Kalman filters

For a comprehensive overview of current development activities and upcoming tasks,
take a look at the [project board](https://github.com/PTB-M4D/PyDynamic/projects/1),
[issues](https://github.com/PTB-M4D/PyDynamic/issues) and
[pull requests](https://github.com/PTB-M4D/PyDynamic/pulls).

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

This work was part of the Joint Support for Impact project
[14SIP08](https://www.euramet.org/research-innovation/search-research-projects/details/project/standards-and-software-to-maximise-end-user-uptake-of-nmi-calibrations-of-dynamic-force-torque-and/)
of the European Metrology Programme for Innovation and Research (EMPIR). The
[EMPIR](http://msu.euramet.org) is jointly funded by the EMPIR participating 
countries within EURAMET and the European Union.

## Disclaimer

This software is developed at Physikalisch-Technische Bundesanstalt (PTB). The
software is made available "as is" free of cost. PTB assumes no responsibility
whatsoever for its use by other parties, and makes no guarantees, expressed or
implied, about its quality, reliability, safety, suitability or any other
characteristic. In no event will PTB be liable for any direct, indirect or
consequential damage arising in connection with the use of this software.

## License

PyDynamic is distributed under the LGPLv3 license except for the module `impinvar.
py` in the package `misc`, which is distributed under the GPLv3 license.
