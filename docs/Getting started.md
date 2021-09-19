# Installation

There is a [quick way](#quick-setup-not-recommended) to get started but we advise to
setup a virtual environment and guide through the process in the section
[Proper Python setup with virtual environment](#proper-python-setup-with-virtual-environment)

## Quick setup (**not recommended**)

If you just want to use the software, the easiest way is to run from your
system's command line

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

### Updating to the newest version

Updates can then be installed via

```shell
pip install --user --upgrade PyDynamic
```

## Proper Python setup with virtual environment  (**recommended**)

The setup described above allows the quick and easy use of PyDynamic, but it also has
its downsides. When working with Python we should rather always work in so-called
virtual environments, in which our project specific dependencies are satisfied
without polluting or breaking other projects' dependencies and to avoid breaking all
our dependencies in case of an update of our Python distribution.

If you are not familiar with [Python virtual environments
](https://docs.python.org/3/glossary.html#term-virtual-environment) you can get the
motivation and an insight into the mechanism in the
[official docs](https://docs.python.org/3/tutorial/venv.html).

### Create a virtual environment and install requirements

Creating a virtual environment with Python built-in tools is easy and explained
in more detail in the
[official docs of Python itself](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments).

It boils down to creating an environment anywhere on your computer, then activate
it and finally install PyDynamic and its dependencies.

#### _venv_ creation and installation in Windows

In your Windows command prompt execute the following:

```shell
> py -3 -m venv LOCAL\PATH\TO\ENVS\PyDynamic_venv
> LOCAL\PATH\TO\ENVS\PyDynamic_venv\Scripts\activate.bat
(PyDynamic_venv) > pip install PyDynamic
```

#### _venv_ creation and installation on Mac and Linux

In your terminal execute the following:

```shell
$ python3 -m venv /LOCAL/PATH/TO/ENVS/PyDynamic_venv
$  /LOCAL/PATH/TO/ENVS/PyDynamic_venv/bin/activate
(PyDynamic_venv) $ pip install PyDynamic
```

### Updating to the newest version

Updates can then be installed on all platforms after activating the virtual environment
via:

```shell
(PyDynamic_venv) $ pip install --upgrade PyDynamic
```
