# PyDynamic
Python package for the analysis of dynamic measurements

The goal of this package is to provide a starting point for users in metrology and related areas who deal with time-dependent, i.e. *dynamic*, measurements.
The software is part of a joint research project of the national metrology institutes from Germany and the UK, i.e. [Physikalisch-Technische Bundesanstalt](http://www.ptb.de/cms/en.html)
and the [National Physical Laboratory](http://www.npl.co.uk).

PyDynamic offers propagation of *uncertainties* for
- application of the discrete Fourier transform and its inverse
- filtering with an FIR or IIR filter with uncertain coefficients
- design of a FIR filter as the inverse of a frequency response with uncertain coefficients
- design on an IIR filter as the inverse of a frequency response with uncertain coefficients
- deconvolution in the frequency domain by division
- multiplication in the frequency domain
- transformation from amplitude and phase to a representation by real and imaginary parts

For the validation of the propagation of uncertainties, the Monte-Carlo method can be applied using a memory-efficient implementation of Monte-Carlo for digital filtering

The documentation for PyDynamic can be found at [http://pydynamic.readthedocs.io](http://pydynamic.readthedocs.io)

![PyDynamic packages](http://mathmet.org/projects/14SIP08/PyDynamic_scheme.png)


### Installation
If you just want to use the software, the easiest way is to run from your system's command line
```
  pip install PyDynamic
```
This will download the latest version from the Python package repository and copy it into your local folder of third-party libraries. Usage in any Python environment on your computer is then possible by
```python
  import PyDynamic
```
or, for example, for the module containing the Fourier domain uncertainty methods:
```python
  from PyDynamic.uncertainty import propagate_DFT
```
Updates can then be installed via
```
  pip install --upgrade PyDynamic
```

For collaboration we recommend using [Github Desktop](https://desktop.github.com) or any other git-compatible version control software and cloning the repository (https://github.com/eichstaedtPTB/PyDynamic.git). In this way, any updates to the software will be highlighted in the version control software and can be applied very easily.

If you have downloaded this software, we would be very thankful for letting us know. You may, for instance, drop an email to one of the authors (e.g. [Sascha Eichstädt](mailto:sascha.eichstaedt@ptb.de) or [Ian Smith](mailto:ian.smith@npl.co.uk) )


### Examples
Uncertainty propagation for the application of a FIR filter with coefficients *b* with which an uncertainty *ub* is associated. The filter input signal is *x* with known 
noise standard deviation *sigma*. The filter output signal is *y* with associated uncertainty *uy*.
```python
    from PyDynamic.uncertainty.propagate_filter import FIRuncFilter
    y, uy = FIRuncFilter(x, sigma, b, ub)    
```


Uncertainty propagation through the application of the discrete Fourier transform (DFT). The time domain signal is *x* with associated squared uncertainty *ux*. The result
of the DFT is the vector *X* of real and imaginary parts of the DFT applied to *x* and the associated uncertainty *UX*.
```python
    from PyDynamic.uncertainty.propagate_DFT import GUM_DFT
    X, UX = GUM_DFT(x, ux)
```


Sequential application of the Monte Carlo method for uncertainty propagation for the case of filtering a time domain signal *x* with an IIR filter *b,a* with uncertainty associated with
the filter coefficients *Uab* and signal noise standard deviation *sigma*. The filter output is the signal *y and the Monte Carlo method calculates point-wise uncertainties *uy* and
coverage intervals *Py* corresponding to the specified percentiles.

```python
    from PyDynamic.uncertainty.propagate_MonteCarlo import SMC
    y, uy, Py = SMC(x, sigma, b, a, Uab, runs=1000, Perc=[0.025,0.975])
```

![PyDynamic Workflow Deconvolution](http://mathmet.org/projects/14SIP08/Deconvolution.png) 

### Roadmap - next steps

1. Extend the existing examples using improved signals and systems.
2. Extend the code to more advanced noise processes and uncertainty evaluation.
3. Add IPython notebooks as a way of documentation.
4. Provide graphical user interfaces for specific use cases (Any suggestions? Let us know!)

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
