# PyDynamic
Python package for the analysis of dynamic measurements

The goal of this package is to provide a starting point for users in metrology and related areas who deal with time-dependent, i.e. *dynamic*, measurements.
The software is part of a joint research project of the national metrology institutes from Germany and the UK, i.e. [Physikalisch-Technische Bundesanstalt](http://www.ptb.de/cms/en.html)
and the [National Physical Laboratory](http://www.npl.co.uk).

### Installation
Since this software is still under active development, we highly recommend to not download the zip-file of the current version only. Instead, we recommend using [Github Desktop](https://desktop.github.com) or any other git-compatible version control software and cloning the repository (https://github.com/eichstaedtPTB/PyDynamic.git). In this way, any updates to the software will be highlighted in the version control software and can be applied very easily.

In the source folder there is a setup script for installation. To run this script, open a command prompt at the location of the 'setup.py' file and run: 'python setup.py install'. This will install the PyDynamic package into your default PYTHONPATH site-packages folder. See 'python setup.py --help' for further instructions. After installation, one can import any module from the PyDynamic package by calling 'import PyDynamic' or similar means, see your favorite Python tutorial for further instructions.

If you have downloaded this software, we would be very thankful for letting us know. You may, for instance, drop an email to one of the authors (e.g. [Sascha Eichstädt](mailto:sascha.eichstaedt@ptb.de) or [Ian Smith](mailto:ian.smith@npl.co.uk) )

### Roadmap - next steps

1. Extend the existing examples using improved signals and systems.
2. Extend the code to more advanced noise processes and uncertainty evaluation.
3. Add IPython notebooks as a way of documentation.
4. Provide graphical user interfaces for specific use cases (Any suggestions? Let us know!)
5. Make package available via `pip`.

##### Acknowledgement
This work is part of the Joint Support for Impact project [14SIP08](http://mathmet.org/projects/14SIP08) of the European Metrology Programme for Innovation and Research (EMPIR). 
The [EMPIR](http://msu.euramet.org) is jointly funded by the EMPIR participating countries within EURAMET and the European Union.

##### Disclaimer
This software was developed at Physikalisch-Technische Bundesanstalt (PTB) and National Physical Laboratory (NPL). 
The software is made available "as is" free of cost. PTB and NPL assume no responsibility whatsoever for its use by other parties, 
and makes no guarantees, expressed or implied, about its quality, reliability, safety, suitability or any other characteristic. 
In no event will PTB and NPL be liable for any direct, indirect or consequential damage arising in connection with the use of this software.

##### License
PyDynamic is distributed under the LGPLv3 license with the exception of the module `impinvar.py` in the package `misc`, which is distributed under the GPLv3 license.
