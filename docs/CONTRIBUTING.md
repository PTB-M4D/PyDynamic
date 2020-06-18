# PyDynamic advice and tips for collaboration

If you want to become active as a developer, we provide all important information
here to make the start as easy as possible. At the same time, the code you produce
should be seamlessly integrable into PyDynamic by aligning your work with the
established workflows from the beginning. This guide should work on all platforms, so
please open an issue or ideally fix the problem and contribute to this guide as a
start, if problems arise.

We will cover

- basic principles
- initial set up
- advised toolset
- coding style
- commit style
- PR workflow
- requirements handling
- and probably more...

## Basic principles

The PyDynamic development process is based on the following guiding principles: 

- support all [major Python versions supported upstream
  ](https://devguide.python.org/#status-of-python-branches).
- actively maintain, ensuring security vulnerabilities or other issues
  are resolved in a timely manner 
- employ state-of-the-art development practices and tools, specifically
  - follow [semantic versioning](https://semver.org/)
  - use [conventional commit messages](https://www.conventionalcommits.org/en/v1.0.0/)
  - consider the PEP8 style guide, wherever feasible

## Get started developing

### Get the code on GitHub and locally

For collaboration we recommend forking the repository as described 
[here](https://help.github.com/en/articles/fork-a-repo), apply the changes and open a
Pull Request on GitHub as described
[here](https://help.github.com/en/articles/creating-a-pull-request). For small
changes this will be a sufficient setup, since this way even the full test suite will
run against your changes. For more comprehensive work, you should read on carefully
and follow the instructions.
   
### Initial development setup

This guide assumes you already have a valid runtime environment for PyDynamic as
described in the [README](../README.md). To start developing, install the required
dependencies for your specific Python version. To find it, activate the desired
virtual environment and execute:

```shell
(PyDynamic_venv) $ python --version
Python 3.8.3
```

Then upgrade/install _pip_ and _pip-tools_ which we use to pin our dependencies to
specific versions:
 
```shell
(PyDynamic_venv) $ pip install --upgrade pip pip-tools
```

From the repository root you can then initially install or at any later time update
all dependencies to the versions we use via _pip-tools_' command 
[`pip-sync`](https://pypi.org/project/pip-tools/#example-usage-for-pip-sync) by
executing e.g. for Python 3.8:

```shell
(PyDynamic_venv) $ pip-sync requirements/dev-requirements-py38.txt requirements/requirements-py38.txt
```

### Advised toolset

We use black to implement our coding style, Sphinx for automated generation of our documentation on ReadTheDocs and pytest backed by hypothesis and coverage as our testing framework, executed by tox. For automated releases we use python-semantic-release in our pipeline on CircleCI. All requirements for contributions are derived from this.