""" Perform tests on the method *uncertainty.propagate_MonteCarlo*"""

import functools

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from PyDynamic.misc.filterstuff import kaiser_lowpass
from PyDynamic.misc.noise import ARMA
from PyDynamic.misc.testsignals import rect
from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.uncertainty.propagate_MonteCarlo import MC, SMC, UMC, UMC_generic

# parameters of simulated measurement
Fs = 100e3  # sampling frequency (in Hz)
Ts = 1 / Fs  # sampling interval length (in s)

# nominal system parameters
fcut = 20e3  # low-pass filter cut-off frequency (6 dB)
L = 100  # filter order
b1 = kaiser_lowpass(L, fcut, Fs)[0]
b2 = kaiser_lowpass(L - 20, fcut, Fs)[0]

# uncertain knowledge: cutoff between 19.5kHz and 20.5kHz
runs = 20
FC = fcut + (2 * np.random.rand(runs) - 1) * 0.5e3

B = np.zeros((runs, L + 1))
for k in range(runs):  # Monte Carlo for filter coefficients of low-pass filter
    B[k, :] = kaiser_lowpass(L, FC[k], Fs)[0]

Ub = make_semiposdef(np.cov(B, rowvar=False))  # covariance matrix of MC result

# simulate input and output signals
nTime = 500
time = np.arange(nTime) * Ts  # time values

# different cases
sigma_noise = 1e-5

# input signal + run methods
x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=sigma_noise)

# evaluate UMC_generic using the following (dummy) functions
sample_shape = (2, 3, 4)
draw_samples = lambda size: np.random.rand(size, *sample_shape)
evaluate_sample = functools.partial(np.mean, axis=1)

UMC_generic_multiprocess_kwargs = {
    "draw_samples": draw_samples,
    "evaluate": evaluate_sample,
    "runs": 10,
    "blocksize": 3,
    "runs_init": 3,
}

UMC_generic_cov_kwargs = {
    "draw_samples": draw_samples,
    "evaluate": evaluate_sample,
    "runs": 5,
    "blocksize": 2,
    "runs_init": 2,
    "return_samples": True,
    "return_histograms": False,
}


def test_MC(visualizeOutput=False):
    # run method
    y, Uy = MC(x, sigma_noise, b1, np.ones(1), Ub, runs=runs, blow=b2)

    assert len(y) == len(x)
    assert Uy.shape == (x.size, x.size)

    if visualizeOutput:
        # visualize input and mean of system response
        plt.plot(time, x)
        plt.plot(time, y)

        # visualize uncertainty of output
        plt.plot(
            time, y - np.sqrt(np.diag(Uy)), linestyle="--", linewidth=1, color="red"
        )
        plt.plot(
            time, y + np.sqrt(np.diag(Uy)), linestyle="--", linewidth=1, color="red"
        )

        plt.show()


def test_MC_non_negative_main_diagonal_covariance():
    _, Uy = MC(x, sigma_noise, b1, np.ones(1), Ub, runs=runs, blow=b2)
    assert np.all(np.diag(Uy) >= 0)


def test_SMC():
    # run method
    y, Uy = SMC(x, sigma_noise, b1, np.ones(1), Ub, runs=runs)

    assert len(y) == len(x)
    assert Uy.shape == y.shape


@pytest.mark.slow
def test_UMC(visualizeOutput=False):
    # run method
    y, Uy, p025, p975, happr = UMC(
        x, b1, [1.0], Ub, blow=b2, sigma=sigma_noise, runs=runs, runs_init=10, nbins=10
    )

    assert len(y) == len(x)
    assert Uy.shape == (x.size, x.size)
    assert p025.shape[1] == len(x)
    assert p975.shape[1] == len(x)
    assert isinstance(happr, dict)

    if visualizeOutput:
        # visualize input and mean of system response
        plt.plot(time, x)
        plt.plot(time, y)

        # visualize uncertainty of output
        plt.plot(
            time, y - np.sqrt(np.diag(Uy)), linestyle="--", linewidth=1, color="red"
        )
        plt.plot(
            time, y + np.sqrt(np.diag(Uy)), linestyle="--", linewidth=1, color="red"
        )

        # visualize central 95%-quantile
        plt.plot(time, p025.T, linestyle=":", linewidth=1, color="gray")
        plt.plot(time, p975.T, linestyle=":", linewidth=1, color="gray")

        # visualize the bin-counts
        key = list(happr.keys())[0]
        for ts, be, bc in zip(
            time, happr[key]["bin-edges"].T, happr[key]["bin-counts"].T
        ):
            plt.scatter(ts * np.ones_like(bc), be[1:], bc)

        plt.show()


def test_UMC_generic_multiprocessing():
    # run UMC
    y, Uy, happr, output_shape = UMC_generic(**UMC_generic_multiprocess_kwargs)
    assert y.size == Uy.shape[0]
    assert Uy.shape == (y.size, y.size)
    assert isinstance(happr, dict)
    assert output_shape == (sample_shape[0], sample_shape[2])


def test_UMC_generic_no_multiprocessing():
    # run without parallel computation
    y, Uy, happr, output_shape = UMC_generic(**UMC_generic_multiprocess_kwargs, n_cpu=1)
    assert y.size == Uy.shape[0]
    assert Uy.shape == (y.size, y.size)
    assert isinstance(happr, dict)
    assert output_shape == (sample_shape[0], sample_shape[2])


def test_UMC_generic_check_sample_shape():
    # run again, but only return all simulations
    y, Uy, happr, output_shape, sims = UMC_generic(
        **UMC_generic_multiprocess_kwargs, return_samples=True
    )
    assert y.size == Uy.shape[0]
    assert Uy.shape == (y.size, y.size)
    assert isinstance(happr, dict)
    assert output_shape == (sample_shape[0], sample_shape[2])
    assert isinstance(sims, dict)
    assert sims["samples"][0].shape == sample_shape
    assert sims["results"][0].shape == output_shape
    assert len(sims["samples"]) == UMC_generic_multiprocess_kwargs["runs"]


def test_UMC_generic_cov_diag():
    # evaluate only diag covariance + return samples (to check against)
    y, Uy, _, _, sims = UMC_generic(**UMC_generic_cov_kwargs, compute_full_covariance=False)

    assert y.size == Uy.shape[0]
    assert Uy.shape == (y.size,)

    y_sims = np.mean(sims["results"], axis=0).flatten()
    Uy_sims = np.diag(
        np.cov(sims["results"].reshape((sims["results"].shape[0], -1)), rowvar=False)
    )

    assert_allclose(y, y_sims)
    assert_allclose(Uy, Uy_sims)


def test_UMC_generic_cov_full():
    # evaluate only diag covariance + return samples (to check against)
    y, Uy, _, _, sims = UMC_generic(**UMC_generic_cov_kwargs, compute_full_covariance=True)

    assert y.size == Uy.shape[0]
    assert Uy.shape == (y.size, y.size)

    y_sims = np.mean(sims["results"], axis=0).flatten()
    Uy_sims = np.cov(
        sims["results"].reshape((sims["results"].shape[0], -1)), rowvar=False
    )

    assert_allclose(y, y_sims)
    assert_allclose(Uy, Uy_sims)


@pytest.mark.slow
def test_compare_MC_UMC():

    np.random.seed(12345)

    y_MC, Uy_MC = MC(x, sigma_noise, b1, np.ones(1), Ub, runs=2 * runs, blow=b2)
    y_UMC, Uy_UMC, _, _, _ = UMC(
        x, b1, [1.0], Ub, blow=b2, sigma=sigma_noise, runs=2 * runs, runs_init=10
    )

    # both methods should yield roughly the same results
    assert_allclose(y_MC, y_UMC, atol=5e-4)
    assert_allclose(Uy_MC, Uy_UMC, atol=5e-4)


def test_noise_ARMA():
    length = 100
    phi = [1 / 3, 1 / 4, 1 / 5]
    theta = [1, -1]

    e = ARMA(length, phi=phi, theta=theta)

    assert len(e) == length
