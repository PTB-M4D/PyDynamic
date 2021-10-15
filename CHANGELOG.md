# Changelog

<!--next-version-placeholder-->

## v1.11.0 (2021-10-15)
### Feature
* **plot_vectors_and_covariances_comparison:** Introduce function to conveniently compare vectors ([`e2b3b0c`](https://github.com/PTB-M4D/PyDynamic/commit/e2b3b0c530fe3970919beeec14c96587a86653af))
* **normalize_vector_or_matrix:** Make normalize_vector_or_matrix() publicly available ([`52b1256`](https://github.com/PTB-M4D/PyDynamic/commit/52b125679472b227612951e869958e1e695dbcfe))
* **is_2d_square_matrix:** Make is_2d_square_matrix() publicly available ([`e303e6b`](https://github.com/PTB-M4D/PyDynamic/commit/e303e6b920c96010e417dec9013e3b6f639466c8))

### Fix
* **__version__:** Reintroduce __version__ variable into PyDynamic/__init__.py ([`0349b09`](https://github.com/PTB-M4D/PyDynamic/commit/0349b09eeff34a2144a8dc86a9bccc6aed0387cd))

### Documentation
* **CONTRIBUTING:** Mention necessity of installing PyDynamic itself for testing ([`1571585`](https://github.com/PTB-M4D/PyDynamic/commit/157158536c1ed7f57f34ba33578ee1cb60716b21))

## v1.10.0 (2021-09-28)
### Feature
* **propagate_DFT:** Make some helpers to check for shapes of inputs publicly available ([`dc97b3f`](https://github.com/PTB-M4D/PyDynamic/commit/dc97b3faddcb854670a9a5f9dcd4ff38efb575d9))

### Fix
* **fit_som:** Apply minor changes to fit_som example to make it executable again ([`0157fc7`](https://github.com/PTB-M4D/PyDynamic/commit/0157fc7f775851359efc6d1a97ca87c9325108be))

## v1.9.2 (2021-09-21)
### Fix
* **PyPI package:** Include examples in packag and make sure all tests pass with delivered version ([`f8326d5`](https://github.com/PTB-M4D/PyDynamic/commit/f8326d52d9dc289b506999365850067001320a3e))

## v1.9.1 (2021-09-15)
### Fix
* **DFT_deconv:** Replace the imaginary part of H by Y's imaginary part in one of the equations ([`a4252dd`](https://github.com/PTB-M4D/PyDynamic/commit/a4252ddd850908467933d7de41f1da99d57d3ea1))

### Documentation
* Introduce Python 3.9 to the docs and actually provide requirements*.txt files ([`19dcef2`](https://github.com/PTB-M4D/PyDynamic/commit/19dcef2f5b1d0516dc9ebf462d5115a5554c8cec))

## v1.9.0 (2021-05-11)
### Feature
* **interp1d_unc:** Add cubic bspline interpolation-kind ([`f0c6d19`](https://github.com/PTB-PSt1/PyDynamic/commit/f0c6d19bad71816f5c6d95803f734e77567931ea))

### Documentation
* **interp1d_unc:** Add example for kind = "cubic" ([`8c2ce38`](https://github.com/PTB-PSt1/PyDynamic/commit/8c2ce38af4cb4a391da9e82c23c5754f494892ad))

## v1.8.0 (2021-04-28)
### Feature
* **propagate_convolution:** Convolution with full covariance propagation ([`299165e`](https://github.com/PTB-PSt1/PyDynamic/commit/299165ef630ef54bcb877e8a3260038805609e4f))

### Documentation
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
