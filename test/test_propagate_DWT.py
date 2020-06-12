"""
Perform test for uncertainty.propagate_DWT
"""

import matplotlib.pyplot as plt
import numpy as np

from PyDynamic.uncertainty.propagate_DWT import dwt, wave_dec, inv_dwt, wave_rec, filter_design, dwt_max_level, wave_dec_realtime

import pywt


def test_filter_design():

    for filter_name in ["db3", "db4", "rbio3.3"]:

        ld, hd, lr, hr = filter_design(filter_name)

        assert isinstance(ld, np.ndarray)
        assert isinstance(hd, np.ndarray)
        assert isinstance(lr, np.ndarray)
        assert isinstance(hr, np.ndarray)


def test_dwt():
    
    for filter_name in ["db3", "db4"]:

        for nx in [20,21,22,23]:

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


def test_idwt():

    for filter_name in ["db3", "db4"]:

        for nc in [20,21,22,23]:

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
            assert 2*c_approx.size - lr.size + 2 == x.size

            # compare to pywt
            r = pywt.idwt(c_approx, c_detail, filter_name, mode="constant")
            assert np.allclose(x, r)


def test_identity_single(make_plots=False):

    for filter_name in ["db3", "db4"]:

        for nx in [20, 21, 22, 23]:

            x = np.linspace(1,nx,nx)  # np.random.randn(nx)
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
            
            if make_plots:
                plt.plot(x)
                plt.plot(xr)
                plt.show()

                plt.plot(Ux)
                plt.plot(Uxr)
                plt.show()


def test_max_level():
    assert dwt_max_level(12, 5) == 1
    assert dwt_max_level(1000, 11) == 6


def test_decomposition():
    for filter_name in ["db2", "db3"]:

        for nx in [20,21]:

            x = np.random.randn(nx)
            Ux = 0.1 * (1 + np.random.random(nx))

            ld, hd, lr, hr = filter_design(filter_name)

            coeffs, Ucoeffs, ol = wave_dec(x, Ux, ld, hd)

            #for c, Uc in zip(coeffs, Ucoeffs):
            #    print(c)
            #    print("-"*60)
            #print("="*60)

            # compare to the output of PyWavelet
            result_pywt = pywt.wavedec(x, filter_name, mode='constant')

            # compare output depth
            assert len(result_pywt) == len(coeffs)
            
            # compare output in detail
            for a, b in zip(result_pywt, coeffs):
                assert len(a) == len(b)
                assert np.allclose(a, b)


def test_decomposition_realtime():
    for filter_name in ["db2", "db3"]:

        for nx in [20,21]:

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
            for x_batch, Ux_batch in zip(np.array_split(x, n_splits), np.array_split(Ux, n_splits)):
                coeffs_b, Ucoeffs_b, ol_b, z_b = wave_dec_realtime(x_batch, Ux_batch, ld, hd, n=2, level_states=z_b)
                coeffs_list.append(coeffs_b)
                Ucoeffs_list.append(Ucoeffs_b)

            coeffs_b = [np.concatenate([coeffs[level] for coeffs in coeffs_list], axis=0) for level in range(len(coeffs_list[0]))]
            Ucoeffs_b = [np.concatenate([Ucoeffs[level] for Ucoeffs in Ucoeffs_list], axis=0) for level in range(len(Ucoeffs_list[0]))]

            # compare output depth
            assert len(coeffs_a) == len(coeffs_b)
            
            # compare output in detail
            for a, b in zip(coeffs_a, coeffs_b):
                assert len(a) == len(b)
                assert np.allclose(a, b)


def test_reconstruction():
    pass


def test_identity_multi():

    for filter_name in ["db3", "db4"]:

        for nx in [20, 21, 203]:

            x = np.linspace(1,nx,nx)  # np.random.randn(nx)
            #Ux = 0.1 * (1 + np.random.random(nx))
            Ux = np.ones((nx))
            Ux[nx//2:] = 2
            Ux = 0.1 * Ux

            ld, hd, lr, hr = filter_design(filter_name)

            # full decomposition
            coeffs, Ucoeffs, ol = wave_dec(x, Ux, ld, hd)

            # full reconstruction
            xr, Uxr = wave_rec(coeffs, Ucoeffs, lr, hr, original_length=ol)

            assert x.size == xr.size
            assert np.allclose(x, xr)
            assert Ux.size == Uxr.size

            #if nx == 203 and filter_name == "db4":
            #    plt.plot(Ux)
            #    plt.plot(Uxr)
            #    plt.show()
