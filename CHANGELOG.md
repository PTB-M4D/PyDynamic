# Changelog

<!--next-version-placeholder-->

## v1.8.0 (2021-04-28)
### Feature
* **propagate_convolution:** Add modes: reflect, nearest, mirror from scipy.ndimage.convolve1D ([`af60c91`](https://github.com/PTB-PSt1/PyDynamic/commit/af60c919d47cacddf7250338bed3b391037cb63f))
* **propagate_convolution:** Convolution with full covariance propagation ([`299165e`](https://github.com/PTB-PSt1/PyDynamic/commit/299165ef630ef54bcb877e8a3260038805609e4f))

### Fix
* **propagate_convolution:** Resolve failing MC comparison ([`afad2b0`](https://github.com/PTB-PSt1/PyDynamic/commit/afad2b0dfe3791d2747684d1a3c938d454e235fe))

### Documentation
* **convolve_unc:** Less technical description and minor changes ([`f743ad8`](https://github.com/PTB-PSt1/PyDynamic/commit/f743ad8895aed391663ca8395d0dcce506d6a19c))
* **propagate_convolution:** Fix module description ([`7604712`](https://github.com/PTB-PSt1/PyDynamic/commit/760471288e7365d2a7e2fb8503574a1a5ca99095))
* **convolve_unc:** Fix typo ([`c1a6228`](https://github.com/PTB-PSt1/PyDynamic/commit/c1a6228df084392b230c2c4b3e6ee74fd5cfec31))
* **convolve_unc:** Remove mentioning of default value ([`5e37fc9`](https://github.com/PTB-PSt1/PyDynamic/commit/5e37fc9b8cad6da6cf8f86ccdec67d8eaa15ac70))
* **convolve_unc:** Adjust seealso ([`478f591`](https://github.com/PTB-PSt1/PyDynamic/commit/478f591b355638fd3ce49a2181ee9aaf54093439))
* **convolve_unc:** Adjust list ([`7fda669`](https://github.com/PTB-PSt1/PyDynamic/commit/7fda669979a4fce7da4318c2f8312c2ef44becbb))
* **propagate_convolution:** Add module description ([`ffafa00`](https://github.com/PTB-PSt1/PyDynamic/commit/ffafa00ea8961bdd4e4414645102f6870063f3d7))
* **DFT_deconv:** Improve wording of docstring for return value ([`e866aa4`](https://github.com/PTB-PSt1/PyDynamic/commit/e866aa41212767f9b24f966bfabf005c9f6cdf39))

## v1.7.0 (2021-02-16)
### Feature
* **FIRuncFilter:** Add FIR filter with full covariance support ([`b937a8b`](https://github.com/PTB-PSt1/PyDynamic/commit/b937a8b979f947df3149aeda38401b90fc522ef4))

### Documentation
* **trimOrPad:** Enhance docstring ([`4c0da58`](https://github.com/PTB-PSt1/PyDynamic/commit/4c0da58e2c9d2bcae212a757f7b80ef65f602cd1))
* **_fir_filter:** Adjust docstring and ValueError ([`090b8f8`](https://github.com/PTB-PSt1/PyDynamic/commit/090b8f832c54e4cb969de70d5825d5000410eacc))

## v1.6.1 (2020-10-29)
### Fix
* Flip theta in uncertainty formulas of FIRuncFilter ([`dd04eea`](https://github.com/PTB-PSt1/PyDynamic/commit/dd04eeace70ce4fe7a81fb432cc117f80af74d4f))
