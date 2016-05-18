# -*- coding: utf-8 -*-
"""
Installation of PyDynamic in Python path
"""

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "PyDynamic",
    version = 1.01,
    author = u"Sascha Eichstädt, Ian Smith",
    author_email = "sascha.eichstaedt@ptb.de",
    description = ("A software package for the analysis of dynamic measurements"),
    license = "LGPLv3",
    keywords = "uncertainty dynamic deconvolution metrology",
	packages = ['deconvolution', 'identification', 'uncertainty', 'misc'],
    long_description = read('README.md'),
	url = 'http://mathmet.org/projects/14SIP08',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: LGPL License"]
)
