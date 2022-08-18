# Changelog

<!--next-version-placeholder-->

## v2.3.0 (2022-08-18)
### Feature
* **LSIIR:** Return the RMS value backwards compatible (see #288) ([`c5c484e`](https://github.com/PTB-M4D/PyDynamic/commit/c5c484e991748641d3596462516e69161c896482))

### Fix
* **test_interpolate:** Ensure interpolation nodes not becoming too small in orders of magnitude ([`f3bf886`](https://github.com/PTB-M4D/PyDynamic/commit/f3bf88662eb8379fb201e610827739c84558913d))
* **test_interpolate:** Avoid only zeros as interpolation nodes in strategy for test case generation ([`2596f24`](https://github.com/PTB-M4D/PyDynamic/commit/2596f245122f9e82132e3cd058e22c6b38291b1c))
* **_compute_stabilized_filter_through_time_delay_iteration:** Correct implementation ([`6286c75`](https://github.com/PTB-M4D/PyDynamic/commit/6286c754d7aad93ca903270b74b9cef15c494207))

**[See all commits in this version](https://github.com/PTB-M4D/PyDynamic/compare/v2.2.0...v2.3.0)**

## v2.2.0 (2022-04-22)
### Feature
* **convolve_unc:** Allow 1D array of stdunc as input ([`ae5335a`](https://github.com/PTB-M4D/PyDynamic/commit/ae5335a54e2452f0e936cb50263865d4b4916a87))

**[See all commits in this version](https://github.com/PTB-M4D/PyDynamic/compare/v2.1.3...v2.2.0)**

## v2.1.3 (2022-04-19)
### Fix
* **test_ARMA:** Increase closeness tolerance ([`e35e536`](https://github.com/PTB-M4D/PyDynamic/commit/e35e536c42966da81930aca05166abe1e04c906a))

**[See all commits in this version](https://github.com/PTB-M4D/PyDynamic/compare/v2.1.2...v2.1.3)**

## v2.1.2 (2022-02-07)
### Fix
* **tools:** Switch to eigs import from scipy.sparse.linalg for scipy>=1.8.0 ([`6618278`](https://github.com/PTB-M4D/PyDynamic/commit/6618278a1f9dda069d433a4a469bbe865a3d54df))

**[See all commits in this version](https://github.com/PTB-M4D/PyDynamic/compare/v2.1.1...v2.1.2)**

## v2.1.1 (2021-12-18)
### Fix
* **LSIIR:** Proper init of final_tau ([`29f2eef`](https://github.com/PTB-M4D/PyDynamic/commit/29f2eefadc05d7cf8affd9727d8afb9b56259737))

### Documentation
* **Signal:** Introduce Signal class into docs ([`0da9b9d`](https://github.com/PTB-M4D/PyDynamic/commit/0da9b9d928688460953ffa3e4b92185c5b45f633))
* **Python 3.10:** Introduce Python 3.10 to the docs ([`a20384a`](https://github.com/PTB-M4D/PyDynamic/commit/a20384abf20cf576d5da6b6f0a5960b78d046093))

**[See all commits in this version](https://github.com/PTB-M4D/PyDynamic/compare/v2.1.0...v2.1.1)**

## v2.1.0 (2021-12-03)
### Feature
* **tools:** Provide convenience functions to prepare input vectors for DFT and filtering ([`6d15922`](https://github.com/PTB-M4D/PyDynamic/commit/6d15922d14d934467710cc0466ad1a21b4d6a066))

### Documentation
* **examples:** Add reference to hydrophone paper ([`3c7880a`](https://github.com/PTB-M4D/PyDynamic/commit/3c7880a977cc35f76bfcf33e60ea3ebc95d56ab5))
* **examples:** Add regularization example inside DFT best practice ([`75f6dcc`](https://github.com/PTB-M4D/PyDynamic/commit/75f6dccc923d084e68872c0afbfce9cc536c15a0))

**[See all commits in this version](https://github.com/PTB-M4D/PyDynamic/compare/v2.0.0...v2.1.0)**

## v2.0.0 (2021-11-05)
### Feature
* Weighted least-squares IIR or FIR filter fit to freq. resp. or reciprocal with uncertainties ([`8aca955`](https://github.com/PTB-M4D/PyDynamic/commit/8aca9554165b805aee82d6081db967d7947b5c1e))
* **DWT:** Add wavelet transform with online-support ([`aed3deb`](https://github.com/PTB-M4D/PyDynamic/commit/aed3deb40f2fa85376ba18a1b9c45b1ffd090036))
* **propagate_DWT:** Add prototype of wave_rec_realtime ([`76ca8df`](https://github.com/PTB-M4D/PyDynamic/commit/76ca8df9e9f4778a8b6b57cefd28523d167cda89))
* **misc:** Add buffer-class for realtime applications ([`d105de2`](https://github.com/PTB-M4D/PyDynamic/commit/d105de2228fee1459c38c2a6ee7596a080496bc4))
* **propagate_DWT:** Return the internal state ([`31fdb19`](https://github.com/PTB-M4D/PyDynamic/commit/31fdb191ea0d49d9b71f824c6733639c3b16edf6))
* **IIRuncFilter:** Always return internal state ([`175357a`](https://github.com/PTB-M4D/PyDynamic/commit/175357a564c7e00a2aa4341eb1e3346c3fe774c0))

### Fix
* **propagate_filter:** Avoid floating point issues with small negative uncertainties via clipping ([`bbe9d13`](https://github.com/PTB-M4D/PyDynamic/commit/bbe9d1334c6ec6c51489b8cb1a19c167ca8c7fa6))
* **FIRuncFilter:** Actually perform shifting for fast computation cases ([`14345c6`](https://github.com/PTB-M4D/PyDynamic/commit/14345c62c848a97df2f791fb99ee2162d17a9f7d))
* **FIRuncFilter:** Output shifting returns expected covariance matrix ([`3c6ca41`](https://github.com/PTB-M4D/PyDynamic/commit/3c6ca4172b1362dd9cd3b0e91ac374dd5f458f3f))
* **propagate_DWT:** Adjust renamed function ([`7978c26`](https://github.com/PTB-M4D/PyDynamic/commit/7978c26cc0dac9dece2f5518d47db6e180fd768a))
* **imports:** Make DWT-methods available from top-level ([`85165a6`](https://github.com/PTB-M4D/PyDynamic/commit/85165a6d034a8ae8ae858d6b791d48dd0e899692))
* **examples:** Remove unsed imports ([`f32d975`](https://github.com/PTB-M4D/PyDynamic/commit/f32d975e23be75fa3387ba861e23ea6433472987))
* **examples:** Remove unused buffer from speed-comparison-filter ([`d02a9f3`](https://github.com/PTB-M4D/PyDynamic/commit/d02a9f36ea67088baeeff0880c468768d38a70d6))
* **IIRuncFilter:** Take sqrt(Ux[0]) in case of kind=corr ([`38bdb99`](https://github.com/PTB-M4D/PyDynamic/commit/38bdb996b7d5fa427097be48ace01ac9896fdccd))
* **IIRuncFilter:** Warn user if Ux is float but kind not diag ([`47e01f5`](https://github.com/PTB-M4D/PyDynamic/commit/47e01f544b7497dee40c51bbc09fe7310066b624))
* **IIRuncFilter:** Use None as default for Uab ([`0e7fd18`](https://github.com/PTB-M4D/PyDynamic/commit/0e7fd18dd94d4610108976aee14322c7feb18531))
* **propagate_filter:** Refine error messages ([`038ef72`](https://github.com/PTB-M4D/PyDynamic/commit/038ef72e4c38f268cbe4dfe645a69743499a4b49))
* **example:** Remove validate_FIRuncFilter ([`76d09a2`](https://github.com/PTB-M4D/PyDynamic/commit/76d09a25c9ec4d1e12f592c2bbd802e819838cdb))
* **example:** Adjust validate_FIRuncFilter ([`7469c91`](https://github.com/PTB-M4D/PyDynamic/commit/7469c913bd0f104fc00b9ddf38ff5ac01ff35e98))
* **examples:** Review validate_DWT_monte_carlo- sort imports- add docstring- fix renamed functions- fix changed signatures\n- apply black ([`0199dfe`](https://github.com/PTB-M4D/PyDynamic/commit/0199dfe02ff8ae322e6304fa955e790739203d63))
* **example:** Enhance realtime_dwt ([`14f54fd`](https://github.com/PTB-M4D/PyDynamic/commit/14f54fd7eb72c5fabb3bd8d63f16a02ea8b2be73))
* **model_estimation:** Introduce new package _model_estimation_ in preparation of deprecations ([`627575c`](https://github.com/PTB-M4D/PyDynamic/commit/627575caf1e066e466b668f81ce019c5a4b59f7f))
* **IIRuncFilter:** Match default kind with FIRuncFilter ([`0a0fdfe`](https://github.com/PTB-M4D/PyDynamic/commit/0a0fdfe7e9bd06f499dd5f5059459c370d7d59e4))
* **propagate_filter:** Fix correlated uncertainty formula ([`70e9375`](https://github.com/PTB-M4D/PyDynamic/commit/70e9375992b6b85524ed80ac99ee0a7d94b4bec6))
* **FIRuncFilter:** Set internal state of lfilter ([`1f60e76`](https://github.com/PTB-M4D/PyDynamic/commit/1f60e76f03f808e7d20821c13a2a2b337ab6d084))
* **validate_DWT_monte_carlo:** Adjust return values of dwt/idwt ([`4dd601b`](https://github.com/PTB-M4D/PyDynamic/commit/4dd601b4732260a9f621cc725e74c5ea3a085991))
* **test_decomposition_realtime:** Adjust concat statement ([`947ed21`](https://github.com/PTB-M4D/PyDynamic/commit/947ed211041c3a12fbf060a72d14f20274145423))
* **wave_dec_realtime:** Missing argument in np.empty ([`583a7b5`](https://github.com/PTB-M4D/PyDynamic/commit/583a7b591b3e32c14038ae267aad7a90fe6ea2fe))
* **idwt:** Remove leftover from debugging ([`7cca19d`](https://github.com/PTB-M4D/PyDynamic/commit/7cca19d53919bb771267f8868535de942fe72db2))
* **idwt:** Adjust boundary conditions ([`b7788ff`](https://github.com/PTB-M4D/PyDynamic/commit/b7788ffc7d6d713eeb7c995fd0f83f2ae78d3f23))
* **test_dwt:** Remove too many unpack values ([`4b52d67`](https://github.com/PTB-M4D/PyDynamic/commit/4b52d6750c307cabeebbcb0c70726534c0a73c00))

### Breaking
* Combine _deconvolution.fit_filter_ and _identification.fit_filter_ into _model_estimation.fit_filter_ and provide access to all functionality via according parameter sets for _model_estimation.fit_filter.LSFIR_ and _model_estimation.fit_filter.LSIIR_. ([`8aca955`](https://github.com/PTB-M4D/PyDynamic/commit/8aca9554165b805aee82d6081db967d7947b5c1e))
* Rename input parameters t and t_new to x and x_new in _PyDynamic.uncertainty.interpolate_  ([`918f5bb`](https://github.com/PTB-M4D/PyDynamic/commit/918f5bb4ecf6239adc2f8e996689b0cef9ca8d9d))
* Rename `fit_sos()` to `fit_som()` because it actually handles second-order models and not
second-order-systems. ([`bc42fd1`](https://github.com/PTB-M4D/PyDynamic/commit/bc42fd142f823feff3c15058ee252b0998541739))

### Documentation
* **README:** Restyle README and generally improve structure of docs ([`1409856`](https://github.com/PTB-M4D/PyDynamic/commit/1409856acf2b576e28f6e2993de58c459baa6243))
* Fix some formatting issues resulting in strange looking or misleading info on ReadTheDocs ([`ab30b4b`](https://github.com/PTB-M4D/PyDynamic/commit/ab30b4bd355b8a176eae022da7cc4f4a826da924))
* **Design of a digital deconvolution filter (FIR type):** Introduce one more example notebook ([`c51b98b`](https://github.com/PTB-M4D/PyDynamic/commit/c51b98b576f1777f3915c995aa32f0c26fad0431))
* **uncertainties:** Integrate DWT-module to docs ([`fb7a99a`](https://github.com/PTB-M4D/PyDynamic/commit/fb7a99a707758862b696b9a018e5aaba21c08df1))
* **propagate_DWT:** Enhance/prettify docstrings ([`1fcfc43`](https://github.com/PTB-M4D/PyDynamic/commit/1fcfc439d97ad554842ed2b019af1d456c391e98))
* **IIRuncFilter:** Minor adjustments to docstring ([`475a754`](https://github.com/PTB-M4D/PyDynamic/commit/475a75453a59990997c79bdf5930757345b9ffe0))
* **propagate_DWT:** Extend module description ([`a007797`](https://github.com/PTB-M4D/PyDynamic/commit/a007797cb2669b731c31bd0785eb0d817aa73bb3))
* **README:** Document in README optional dependency installation for Jupyter Notebooks ([`a59f98d`](https://github.com/PTB-M4D/PyDynamic/commit/a59f98dec11131b19679beaa44366fea16629c9f))
* **propagate_filter:** Fix IIRuncFilter docstring ([`e2bd085`](https://github.com/PTB-M4D/PyDynamic/commit/e2bd085121a3747aa407edec66c9b7d819f05161))
* **propagate_filter:** Mention FIR and IIR difference ([`f6dcd4e`](https://github.com/PTB-M4D/PyDynamic/commit/f6dcd4efabbc58ad258616bced1ce0369863b751))
* **examples:** Move validation script to examples ([`abc0fd9`](https://github.com/PTB-M4D/PyDynamic/commit/abc0fd98f32e00cfb9df0c786fce5f36b98f2798))
* **examples:** Include errorbars instead of lines ([`76d978e`](https://github.com/PTB-M4D/PyDynamic/commit/76d978eaa5bf8b02e0ac00595b50856f8cc5983d))
* **examples:** Use latex font and adjust naming ([`57f4c83`](https://github.com/PTB-M4D/PyDynamic/commit/57f4c83b6b42e831d978c8eb21a6a27deca8fa24))
* **examples:** Higher uncertainty, tight layout ([`58401c3`](https://github.com/PTB-M4D/PyDynamic/commit/58401c3c4e981ab002156c8f5fefea78546a9e36))
* **examples:** Refining plot output ([`3d0e64c`](https://github.com/PTB-M4D/PyDynamic/commit/3d0e64ca5f34590583b3f1dfb5956be11b27e730))
* **examples:** Calculate and highlight 10 biggest coeffs ([`7e754af`](https://github.com/PTB-M4D/PyDynamic/commit/7e754afc1f6853fa1b0287716a784b3cec7f9f74))
* **examples:** Change order and appearance of plots ([`3138d51`](https://github.com/PTB-M4D/PyDynamic/commit/3138d517143d4855daa83e7f4e7b51ebf345b3a1))
* **examples:** Realtime_dwt with multi-level-decomposition ([`6d48ba7`](https://github.com/PTB-M4D/PyDynamic/commit/6d48ba74d3cc0105dd8272a112b0dfdadf3dcea7))
* **examples:** Plot detail coefficients ([`53ca6f5`](https://github.com/PTB-M4D/PyDynamic/commit/53ca6f5402bfe01152ec954ff1304560978301d0))
* **examples:** Apply dwt to signal ([`93edef5`](https://github.com/PTB-M4D/PyDynamic/commit/93edef5f3eaaf31e48e249d5d7b90349b38c1359))
* **examples:** Add script to examine realtime Wavelet ([`eaf13e7`](https://github.com/PTB-M4D/PyDynamic/commit/eaf13e78bcb28169c9ec95adf8707db9f7a59a02))
* **IIRuncFilter:** Fix wrong formula reference ([`0999569`](https://github.com/PTB-M4D/PyDynamic/commit/0999569d6bb023ddd34cba12686b21637e374b93))
* **propagate_filter:** Adjust return values of IIRuncFilter ([`02a2350`](https://github.com/PTB-M4D/PyDynamic/commit/02a235000c80b638a30dfb077b87d78492117a05))
* **IIRuncFilter:** Describe non-use of b, a, Uab if state ([`0889475`](https://github.com/PTB-M4D/PyDynamic/commit/0889475082976181968c3d02434919bfce2ce10f))
* **propagate_filter:** Enhance specification of "kind" ([`ee2062d`](https://github.com/PTB-M4D/PyDynamic/commit/ee2062dc4687208175ade5028727b6ec14344d75))

**[See all commits in this version](https://github.com/PTB-M4D/PyDynamic/compare/v1.11.1...v2.0.0)**

## v1.11.1 (2021-10-20)
### Fix
* **IIRuncFilter:** Introduce a missing import of scipy.signal.tf2ss ([`17fe115`](https://github.com/PTB-M4D/PyDynamic/commit/17fe115301e048d68a9fd27087cf9739fd3b5bd1))

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
