# Advices and tips for contributors of PyDynamic

If you want to become active as developer, we provide all important information
here to make the start as easy as possible. At the same time, the code you produce
should be seamlessly integrable into PyDynamic by aligning your work with the
established workflows from the beginning. This guide should work on all platforms, so
please open an issue or ideally fix the problem and contribute to this guide as a
start, if problems arise.

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
changes it will be sufficient to just apply your changes on GitHub and send the PR
right away. For more comprehensive work, you should read on carefully, follow the
instructions.
   
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

We use [_black_](https://pypi.org/project/black/) to implement our coding style,
[_Sphinx_](https://pypi.org/project/Sphinx/) for automated generation of [our
 documentation on ReadTheDocs](https://pydynamic.readthedocs.io/en/latest/) and
[_pytest_](https://pypi.org/project/pytest/) backed by
[_hypothesis_](https://pypi.org/project/hypothesis/) and
[_coverage_](https://pypi.org/project/coverage/) as our testing framework, managed by
[_tox_](https://pypi.org/project/tox/). For automated releases we use
[_python-semantic-release_](https://github.com/relekang/python-semantic-release) in
[our pipeline on _CircleCI_](https://app.circleci.com/pipelines/github/PTB-PSt1/PyDynamic)
. All requirements for contributions are derived from this. If you followed the
steps for the [initial development setup](#Initial-development-setup) you have
everything at your hands.

### Coding style

As long as the readability of mathematical formulations is not impaired, our code
should follow the guidelines of [PEP8](https://www.python.org/dev/peps/pep-0008/) and
remain uniformly formatted.  For this purpose we use the Python package
[_black_](https://pypi.org/project/black/). It is very easy to handle and [can be
 integrated into most common IDEs](https://github.com/psf/black#editor-integration),
so that it is automatically applied.

### Commit style

In order for the [semantic versioning](https://semver.org/) release automation and the
[generation of a changelog](https://github.com/PTB-PSt1/PyDynamic/releases/tag/v1.4.0)
from the commit messages to work, it is necessary to use 
[conventional commit messages](https://www.conventionalcommits.org/en/v1.0.0/#summary)
. As a result, parts of the commit messages appear in the changelog of the subsequent
release and the respective commit is linked. We use the following types:

- _feat_: for commits that introduce new features 
- _docs_: for commits that contribute significantly to documentation
- _fix_: commits in which bugs are fixed
- _build_: Commits that affect packaging
- _ci_: Commits that affect the CI pipeline
- _test_: Commits that apply significant changes to tests
- _chore_: Commits that affect other non-PyDynamic components (e.g. ReadTheDocs, Git
, ... )
- _revert_: commits, which undo previous commits using `git revert`
- _refactor_: commits that merely reformulate, rename or similar
- _style_: commits, which essentially make changes to line breaks and whitespace
- _wip_: Commits which, as part of a whole, are not usually recognizable as one of the
  above-mentioned types until later, usually during a PR merge.  The merge commit is
  then marked as the corresponding type.

###  Testing

We strive to increase [our code coverage](https://codecov.io/gh/PTB-PSt1/PyDynamic) with 
every change introduced. This requires that every new feature and every change to 
existing features is accompanied by appropriate _pytest_ testing, which checks the basic
components for correctness and, if necessary, the integration into the big picture.
For this purpose, it is usually sufficient to create
[appropriately named](https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery)
methods in one of the existing modules in the subfolder test or, if necessary, to add
a new module that is also appropriately named.
