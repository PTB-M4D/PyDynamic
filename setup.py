# -*- coding: utf-8 -*-
"""
Install PyDynamic in Python path.
"""

import os
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install

# Get release version from PyDynamic __init__.py
from PyDynamic import __version__ as VERSION


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def readme():
    """Print long description"""
    with open('README.md') as f:
        return f.read()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'Verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: " \
                   "{1}".format(tag, VERSION)
            sys.exit(info)


setup(
    name="PyDynamic",
    version=VERSION,
    description="A software package for the analysis of dynamic measurements",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/eichstaedtPTB/PyDynamic',
    author=u"Sascha Eichstädt, Ian Smith, Thomas Bruns, Björn Ludwig, "
           u"Maximilian Gruber",
    author_email="sascha.eichstaedt@ptb.de",
    keywords="uncertainty dynamic deconvolution metrology",
    packages=find_packages(exclude=["test"]),
    install_requires=[
        'ipython>=7.6.0',
        'jupyter==1.0.0',
        'matplotlib==3.1.1',
        'numpy==1.16.4',
        'pandas==0.24.2',
        'scipy==1.3.0',
        'sympy==1.4'
    ],
    python_requires='>=3',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU Lesser General Public License v3 ("
        "LGPLv3)",
        "Programming Language :: Python :: 3"],
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
