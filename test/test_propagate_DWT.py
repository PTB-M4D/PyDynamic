"""
Perform test for uncertainty.propagate_DWT
"""

import numpy as np
import pywt

from PyDynamic.uncertainty.propagate_DWT import (
    dwt,
    dwt_max_level,
    filter_design,
    inv_dwt,
    wave_dec,
    wave_dec_realtime,
    wave_rec,
)


def test_filter_design():
    """Check if connection to PyWavelets works as expected."""

    for filter_name in ["db3", "db4", "rbio3.3"]:

        ld, hd, lr, hr = filter_design(filter_name)

        assert isinstance(ld, np.ndarray)
        assert isinstance(hd, np.ndarray)
        assert isinstance(lr, np.ndarray)
        assert isinstance(hr, np.ndarray)


def test_dwt():
    """Compare :func:`dwt` to the implementation of :mod:`PyWavelets`"""

    for filter_name in ["db3", "db4"]:

        for nx in [20, 21, 22, 23]:

            x = np.random.randn(nx)
            Ux = 0.1 * (1 + np.random.random(nx))

            ld, hd, lr, hr = filter_design(filter_name)

            # execute single level DWT
            y1, Uy1, y2, Uy2, _ = dwt(x, Ux, ld, hd)

            # all output has same length
            assert y1.size == y2.size
            assert y1.size == Uy1.size
            assert Uy1.size == Uy2.size

            # output is half the length of (input + filter - 1)
            assert (x.size + ld.size - 1) // 2 == y1.size

            # compare to pywt
            ca, cd = pywt.dwt(x, filter_name, mode="constant")
            assert ca.size == y1.size
            assert cd.size == y2.size
            assert np.allclose(ca, y1)
            assert np.allclose(cd, y2)


def test_inv_dwt():
    """Compare :func:`inv_dwt` to the implementation of :mod:`PyWavelets`"""

    for filter_name in ["db3", "db4"]:

        for nc in [20, 21, 22, 23]:

            c_approx = np.random.randn(nc)
            Uc_approx = 0.1 * (1 + np.random.random(nc))
            c_detail = np.random.randn(nc)
            Uc_detail = 0.1 * (1 + np.random.random(nc))

            ld, hd, lr, hr = filter_design(filter_name)

            # execute single level DWT
            x, Ux, _ = inv_dwt(c_approx, Uc_approx, c_detail, Uc_detail, lr, hr)

            # all output has same length
            assert x.size == Ux.size

            # output double the size of input minus filter
            assert 2 * c_approx.size - lr.size + 2 == x.size

            # compare to pywt
            r = pywt.idwt(c_approx, c_detail, filter_name, mode="constant")
            assert np.allclose(x, r)


def test_identity_single(make_plots=False):
    """Test that x = inv_dwt(dwt(x)) for a single level DWT"""

    for filter_name in ["db3", "db4"]:

        for nx in [20, 21, 22, 23]:

            x = np.linspace(1, nx, nx)  # np.random.randn(nx)
            Ux = 0.1 * (1 + np.random.random(nx))

            ld, hd, lr, hr = filter_design(filter_name)

            # single decomposition
            y_approx, U_approx, y_detail, U_detail, _ = dwt(x, Ux, ld, hd)

            # single reconstruction
            xr, Uxr, _ = inv_dwt(y_approx, U_approx, y_detail, U_detail, lr, hr)

            if x.size % 2 == 0:
                assert x.size == xr.size
                assert Ux.size == Uxr.size
                assert np.allclose(x, xr)
            else:
                assert x.size + 1 == xr.size
                assert Ux.size + 1 == Uxr.size
                assert np.allclose(x, xr[:-1])


def test_max_level():
    assert dwt_max_level(12, 5) == 1
    assert dwt_max_level(1000, 11) == 6


def test_wave_dec():
    """Compare :func:`wave_dec` to the implementation of :mod:`PyWavelets`"""
    for filter_name in ["db2", "db3"]:

        for nx in [20, 21]:

            x = np.random.randn(nx)
            Ux = 0.1 * (1 + np.random.random(nx))

            ld, hd, lr, hr = filter_design(filter_name)

            coeffs, Ucoeffs, ol = wave_dec(x, Ux, ld, hd)

            # compare to the output of PyWavelet
            result_pywt = pywt.wavedec(x, filter_name, mode="constant")

            # compare output depth
            assert len(result_pywt) == len(coeffs)

            # compare output in detail
            for a, b in zip(result_pywt, coeffs):
                assert len(a) == len(b)
                assert np.allclose(a, b)


def test_decomposition_realtime():
    """Check if repetitive calls to :func:`wave_dec_realtime` yield the same
    result as a single call to the same function. (Because of different treatment 
    of initial conditions, this can't be directly compared to :func:`wave_dec`.)
    """
    for filter_name in ["db2", "db3"]:

        for nx in [20, 21]:

            x = np.random.randn(nx)
            Ux = 0.1 * (1 + np.random.random(nx))

            ld, hd, lr, hr = filter_design(filter_name)

            # run x all at once
            coeffs_a, Ucoeffs_a, ol_a, z_a = wave_dec_realtime(x, Ux, ld, hd, n=2)

            # slice x into smaller chunks and process them in batches
            # this tests the internal state options
            coeffs_list = []
            Ucoeffs_list = []
            z_b = None
            n_splits = 3
            for x_batch, Ux_batch in zip(
                np.array_split(x, n_splits), np.array_split(Ux, n_splits)
            ):
                coeffs_b, Ucoeffs_b, ol_b, z_b = wave_dec_realtime(
                    x_batch, Ux_batch, ld, hd, n=2, level_states=z_b
                )
                coeffs_list.append(coeffs_b)
                Ucoeffs_list.append(Ucoeffs_b)

            coeffs_b = [
                np.concatenate([coeffs[level] for coeffs in coeffs_list], axis=0)
                for level in range(len(coeffs_list[0]))
            ]
            Ucoeffs_b = [
                np.concatenate([Ucoeffs[level] for Ucoeffs in Ucoeffs_list], axis=0)
                for level in range(len(Ucoeffs_list[0]))
            ]

            # compare output depth
            assert len(coeffs_a) == len(coeffs_b)
            assert len(Ucoeffs_a) == len(Ucoeffs_b)

            # compare output in detail
            for a, b in zip(coeffs_a, coeffs_b):
                assert len(a) == len(b)
                assert np.allclose(a, b)

            # compare output uncertainty in detail
            for a, b in zip(Ucoeffs_a, Ucoeffs_b):
                assert len(a) == len(b)
                assert np.allclose(a, b)


def test_wave_rec():
    """Compare :func:`wave_rec` to the implementation of :mod:`PyWavelets`"""
    for filter_name in ["db2", "db3"]:

        for nx in [20, 21]:
            # generate required coeffs-structure
            coeff_lengths = [
                len(c) for c in pywt.wavedec(np.zeros(nx), filter_name, mode="constant")
            ]

            coeffs = []
            Ucoeffs = []
            for i in coeff_lengths:
                coeffs.append(np.random.random(i))
                Ucoeffs.append(np.random.random(i))

            # define a filter
            ld, hd, lr, hr = filter_design(filter_name)

            x, Ux = wave_rec(coeffs, Ucoeffs, lr, hr)

            # compare to the output of PyWavelet
            result_pywt = pywt.waverec(coeffs, filter_name, mode="constant")

            # compare output of both methods
            assert len(result_pywt) == len(x)
            assert np.allclose(result_pywt, x)


def test_identity_multi():
    """Test that x = inv_dwt(dwt(x)) for a multi level DWT"""
    for filter_name in ["db3", "db4"]:

        for nx in [20, 21, 203]:

            x = np.linspace(1, nx, nx)
            Ux = np.ones((nx))
            Ux[nx // 2 :] = 2
            Ux = 0.1 * Ux

            ld, hd, lr, hr = filter_design(filter_name)

            # full decomposition
            coeffs, Ucoeffs, ol = wave_dec(x, Ux, ld, hd)

            # full reconstruction
            xr, Uxr = wave_rec(coeffs, Ucoeffs, lr, hr, original_length=ol)

            assert x.size == xr.size
            assert np.allclose(x, xr)
            assert Ux.size == Uxr.size
