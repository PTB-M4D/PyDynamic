# Installation

There is a [quick way](#quick-setup-not-recommended) to get started, but we advise 
setting up a virtual environment and guide through the process in the section
[Proper Python setup with virtual environment
](#proper-python-setup-with-virtual-environment-recommended)

## Quick setup (**not recommended**)

If you just want to use the software, the easiest way is to run from your
system's command line

```shell
pip install --user PyDynamic
```

This will download the latest version from the Python package repository
and copy it into your local folder of third-party libraries. Note that
PyDynamic runs with **Python versions 3.7 to 3.10**. Usage in any Python
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

### Set up a virtual environment

If you are not familiar with [Python virtual environments
](https://docs.python.org/3/glossary.html#term-virtual-environment) you can get the
motivation and an insight into the mechanism in the
[official docs](https://docs.python.org/3/tutorial/venv.html).

You have the option to set up PyDynamic using _Anaconda_, if you already have it 
installed, or use the Python built-in tool `venv`. The commands differ slightly 
between [Windows](#create-a-venv-python-environment-on-windows) and [Mac/Linux
](#create-a-venv-python-environment-on-mac-linux) or if you use [Anaconda
](#create-an-anaconda-python-environment).

#### Create a `venv` Python environment on Windows

In your Windows PowerShell execute the following to set up a virtual environment in
a folder of your choice.

```shell
PS C:> cd C:\LOCAL\PATH\TO\ENVS
PS C:\LOCAL\PATH\TO\ENVS> py -3 -m venv PyDynamic_venv
PS C:\LOCAL\PATH\TO\ENVS> PyDynamic_venv\Scripts\activate
```

Proceed to [the next step](#install-pydynamic-via-pip).

#### Create a `venv` Python environment on Mac & Linux

In your terminal execute the following to set up a virtual environment in a folder
of your choice.

```shell
$ cd /LOCAL/PATH/TO/ENVS
$ python3 -m venv PyDynamic_venv
$ source PyDynamic_venv/bin/activate
```

Proceed to [the next step](#install-pydynamic-via-pip).

#### Create an Anaconda Python environment

To get started with your present *Anaconda* installation just go to *Anaconda
prompt* and execute

```shell
$ cd /LOCAL/PATH/TO/ENVS
$ conda env create --file /LOCAL/PATH/TO/PyDynamic/requirements/environment.yml 
```

That's it!

### Install PyDynamic via `pip`

Once you activated your virtual environment, you can install PyDynamic via:

```shell
pip install PyDynamic
```

```shell
Collecting PyDynamic
[...]
Successfully installed PyDynamic-[...] [...]
```

That's it!

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