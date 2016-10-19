# -*- coding: utf-8 -*-
"""
Installation of PyDynamic in Python path
"""

import os
from setuptools import setup, find_packages
from PyDynamic import __version__ as version

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "PyDynamic",
    version = version,
    author = u"Sascha Eichstädt, Ian Smith",
    author_email = "sascha.eichstaedt@ptb.de",
    description = ("A software package for the analysis of dynamic measurements"),
    license = "LGPLv3",
    keywords = "uncertainty dynamic deconvolution metrology",
	packages = find_packages(exclude=["test"]),
    long_description = "Python package for the analysis of dynamic measurements\n The goal of this package is to provide a starting point for users in metrology and related areas who deal with time-dependent, i.e. *dynamic*, measurements.\nThe software is part of a joint research project of the national metrology institutes from Germany and the UK, i.e. [Physikalisch-Technische Bundesanstalt](http://www.ptb.de/cms/en.html) and the [National Physical Laboratory](http://www.npl.co.uk).\n",
	url = 'https://github.com/eichstaedtPTB/PyDynamic',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)"]
)
