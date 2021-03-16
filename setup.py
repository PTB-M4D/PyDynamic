# -*- coding: utf-8 -*-
"""Install PyDynamic in Python path and provide all packaging metadata."""

from os import path

from setuptools import find_packages, setup

# Get release version from PyDynamic __init__.py
from PyDynamic import __version__ as VERSION


def get_readme():
    """Get README.md's content"""
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
        return f.read()


setup(
    metadata_version="2.1",
    name="PyDynamic",
    version=VERSION,
    description="A software package for the analysis of dynamic measurements",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url="https://ptb-pst1.github.io/PyDynamic/",
    download_url="https://github.com/PTB-PSt1/PyDynamic/releases/download/v{0}/"
    "PyDynamic-{0}.tar.gz".format(VERSION),
    author=u"Sascha Eichstädt, Maximilian Gruber, Björn Ludwig, Thomas Bruns, "
    "Martin Weber",
    author_email="sascha.eichstaedt@ptb.de",
    keywords="uncertainty dynamic deconvolution metrology",
    packages=find_packages(exclude=["test"]),
    project_urls={
        "Documentation": "https://pydynamic.readthedocs.io/en/v{}/".format(VERSION),
        "Source": "https://github.com/PTB-PSt1/PyDynamic/tree/v{}/".format(VERSION),
        "Tracker": "https://github.com/PTB-PSt1/PyDynamic/issues",
    },
    install_requires=["matplotlib", "numpy", "pandas", "scipy"],
    # This allow to do "pip install PyDynamic[examples]" and get the dependencies to
    # execute the Jupyter Notebook examples.
    extras_require={"examples": ["notebook"],},
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or "
        "later (LGPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
)
