import os
import pathlib

import numpy as np
import pytest
import scipy.signal as dsp
from hypothesis import given, HealthCheck, settings
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose, assert_almost_equal

from PyDynamic.misc.filterstuff import kaiser_lowpass
from PyDynamic.misc.SecondOrderSystem import sos_phys2filter
from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.model_estimation.fit_filter import (
    invLSFIR,
    invLSFIR_unc,
    invLSFIR_uncMC,
    LSFIR,
)
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter
from .conftest import hypothesis_dimension


@pytest.fixture(scope="module")
def random_number_generator():
    return np.random.default_rng(1)


@pytest.fixture(scope="module")
def measurement_system():
    f0 = 36e3
    S0 = 0.4
    delta = 0.01
    return {"f0": f0, "S0": S0, "delta": delta}


@pytest.fixture(scope="module")
def sampling_frequency():
    return 500e3


@pytest.fixture(scope="module")
def frequencies():
    return np.linspace(0, 120e3, 200)


@pytest.fixture(scope="module")
def monte_carlo(
    measurement_system,
    random_number_generator,
    sampling_frequency,
    frequencies,
    complex_frequency_response,
):
    udelta = 0.1 * measurement_system["delta"]
    uS0 = 0.001 * measurement_system["S0"]
    uf0 = 0.01 * measurement_system["f0"]

    runs = 10000
    MCS0 = (
        measurement_system["S0"] + random_number_generator.standard_normal(runs) * uS0
    )
    MCd = (
        measurement_system["delta"]
        + random_number_generator.standard_normal(runs) * udelta
    )
    MCf0 = (
        measurement_system["f0"] + random_number_generator.standard_normal(runs) * uf0
    )
    HMC = np.zeros((runs, len(frequencies)), dtype=complex)
    for k in range(runs):
        bc_, ac_ = sos_phys2filter(MCS0[k], MCd[k], MCf0[k])
        b_, a_ = dsp.bilinear(bc_, ac_, sampling_frequency)
        HMC[k, :] = dsp.freqz(b_, a_, 2 * np.pi * frequencies / sampling_frequency)[1]

    H = np.r_[np.real(complex_frequency_response), np.imag(complex_frequency_response)]
    assert_allclose(
        H,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_H.npz",
            ),
        )["H"],
    )
    uAbs = np.std(np.abs(HMC), axis=0)
    assert_allclose(
        uAbs,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_uAbs.npz",
            ),
        )["uAbs"],
        rtol=3.5e-2,
    )
    uPhas = np.std(np.angle(HMC), axis=0)
    assert_allclose(
        uPhas,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_uPhas.npz",
            ),
        )["uPhas"],
        rtol=4.3e-2,
    )
    UH = np.cov(np.hstack((np.real(HMC), np.imag(HMC))), rowvar=False)
    UH = make_semiposdef(UH)
    assert_allclose(
        UH,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_UH.npz",
            ),
        )["UH"],
        atol=1,
    )
    return {"H": H, "uAbs": uAbs, "uPhas": uPhas, "UH": UH}


@pytest.fixture(scope="module")
def digital_filter(measurement_system, sampling_frequency):
    # transform continuous system to digital filter
    bc, ac = sos_phys2filter(
        measurement_system["S0"], measurement_system["delta"], measurement_system["f0"]
    )
    assert_almost_equal(bc, [20465611686.098896])
    assert_allclose(ac, np.array([1.00000000e00, 4.52389342e03, 5.11640292e10]))
    b, a = dsp.bilinear(bc, ac, sampling_frequency)
    assert_allclose(
        b, np.array([0.019386043211510096, 0.03877208642302019, 0.019386043211510096])
    )
    assert_allclose(a, np.array([1.0, -1.7975690550957188, 0.9914294872108197]))
    return {"b": b, "a": a}


@pytest.fixture(scope="module")
def complex_frequency_response(
    measurement_system, sampling_frequency, frequencies, digital_filter
):
    Hf = dsp.freqz(
        digital_filter["b"],
        digital_filter["a"],
        2 * np.pi * frequencies / sampling_frequency,
    )[1]
    assert_allclose(
        Hf,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_Hf.npz",
            ),
        )["Hf"],
    )
    return Hf


@pytest.fixture(scope="module")
def simulated_measurement_input_and_output(
    sampling_frequency, digital_filter, random_number_generator
):
    Ts = 1 / sampling_frequency
    time = np.arange(0, 4e-3 - Ts, Ts)
    # x = shocklikeGaussian(time, t0 = 2e-3, sigma = 1e-5, m0=0.8)
    m0 = 0.8
    sigma = 1e-5
    t0 = 2e-3
    x = (
        -m0
        * (time - t0)
        / sigma
        * np.exp(0.5)
        * np.exp(-((time - t0) ** 2) / (2 * sigma ** 2))
    )
    assert_allclose(
        x,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_x.npz",
            ),
        )["x"],
    )
    y = dsp.lfilter(digital_filter["b"], digital_filter["a"], x)
    noise = 1e-3
    yn = y + random_number_generator.standard_normal(np.size(y)) * noise

    return {"time": time, "x": x, "yn": yn, "noise": noise}


@pytest.fixture(scope="module")
def invLSFIR_unc_filter_fit(monte_carlo, frequencies, sampling_frequency):
    N = 12
    tau = N // 2
    bF, UbF = invLSFIR_unc(
        monte_carlo["H"],
        monte_carlo["UH"],
        N,
        tau,
        frequencies,
        sampling_frequency,
    )
    assert np.all(np.linalg.eigvals(UbF) >= 0)
    assert_allclose(
        bF,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_bF.npz",
            ),
        )["bF"],
    )
    assert_allclose(
        UbF,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_UbF.npz",
            ),
        )["UbF"],
        rtol=3e-1,
    )
    return {"bF": bF, "UbF": UbF, "N": N, "tau": tau}


@pytest.fixture(scope="module")
def fir_low_pass(measurement_system, sampling_frequency):
    fcut = measurement_system["f0"] + 10e3
    low_order = 100
    blow, lshift = kaiser_lowpass(low_order, fcut, sampling_frequency)
    assert_allclose(
        blow,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_blow.npz",
            ),
        )["blow"],
    )
    return {"blow": blow, "lshift": lshift}


@pytest.fixture(scope="module")
def shift(
    simulated_measurement_input_and_output,
    invLSFIR_unc_filter_fit,
    fir_low_pass,
):
    shift = (len(invLSFIR_unc_filter_fit["bF"]) - 1) // 2 + fir_low_pass["lshift"]
    assert_allclose(
        shift,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_shift.npz",
            ),
        )["shift"],
    )
    return shift


@pytest.fixture(scope="module")
def fir_unc_filter(
    shift,
    simulated_measurement_input_and_output,
    invLSFIR_unc_filter_fit,
    fir_low_pass,
):
    xhat, Uxhat = FIRuncFilter(
        simulated_measurement_input_and_output["yn"],
        simulated_measurement_input_and_output["noise"],
        invLSFIR_unc_filter_fit["bF"],
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_UbF.npz",
            ),
        )["UbF"],
        shift,
        fir_low_pass["blow"],
    )
    return {"xhat": xhat, "Uxhat": Uxhat}


def test_digital_deconvolution_FIR_example_figure_1(
    frequencies, complex_frequency_response, monte_carlo
):
    plt.figure(figsize=(16, 8))
    plt.errorbar(
        frequencies * 1e-3,
        np.abs(complex_frequency_response),
        monte_carlo["uAbs"],
        fmt=".",
    )
    plt.title("measured amplitude spectrum with associated uncertainties")
    plt.xlim(0, 50)
    plt.xlabel("frequency / kHz", fontsize=20)
    plt.ylabel("magnitude / au", fontsize=20)
    # plt.show()


def test_digital_deconvolution_FIR_example_figure_2(
    frequencies, complex_frequency_response, monte_carlo
):
    plt.figure(figsize=(16, 8))
    plt.errorbar(
        frequencies * 1e-3,
        np.angle(complex_frequency_response),
        monte_carlo["uPhas"],
        fmt=".",
    )
    plt.title("measured phase spectrum with associated uncertainties")
    plt.xlim(0, 50)
    plt.xlabel("frequency / kHz", fontsize=20)
    plt.ylabel("phase / rad", fontsize=20)
    # plt.show()


def test_digital_deconvolution_FIR_example_figure_3(
    simulated_measurement_input_and_output,
):
    plt.figure(figsize=(16, 8))
    plt.plot(
        simulated_measurement_input_and_output["time"] * 1e3,
        simulated_measurement_input_and_output["x"],
        label="system input signal",
    )
    plt.plot(
        simulated_measurement_input_and_output["time"] * 1e3,
        simulated_measurement_input_and_output["yn"],
        label="measured output " "signal",
    )
    plt.legend(fontsize=20)
    plt.xlim(1.8, 4)
    plt.ylim(-1, 1)
    plt.xlabel("time / ms", fontsize=20)
    plt.ylabel(r"signal amplitude / $m/s^2$", fontsize=20)
    # plt.show()


def test_digital_deconvolution_FIR_example_figure_4(
    invLSFIR_unc_filter_fit, monte_carlo, frequencies, sampling_frequency
):
    plt.figure(figsize=(16, 8))
    plt.errorbar(
        range(len(invLSFIR_unc_filter_fit["bF"])),
        invLSFIR_unc_filter_fit["bF"],
        np.sqrt(np.diag(invLSFIR_unc_filter_fit["UbF"])),
        fmt="o",
    )
    plt.xlabel("FIR coefficient index", fontsize=20)
    plt.ylabel("FIR coefficient value", fontsize=20)
    # plt.show()


def test_digital_deconvolution_FIR_example_figure_5(
    invLSFIR_unc_filter_fit,
    frequencies,
    sampling_frequency,
    complex_frequency_response,
    fir_low_pass,
):
    plt.figure(figsize=(16, 10))
    HbF = (
        dsp.freqz(
            invLSFIR_unc_filter_fit["bF"],
            1,
            2 * np.pi * frequencies / sampling_frequency,
        )[1]
        * dsp.freqz(
            fir_low_pass["blow"], 1, 2 * np.pi * frequencies / sampling_frequency
        )[1]
    )
    assert_allclose(
        HbF,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_HbF.npz",
            ),
        )["HbF"],
    )
    plt.semilogy(
        frequencies * 1e-3,
        np.abs(complex_frequency_response),
        label="measured frequency response",
    )
    plt.semilogy(frequencies * 1e-3, np.abs(HbF), label="inverse filter")
    plt.semilogy(
        frequencies * 1e-3,
        np.abs(complex_frequency_response * HbF),
        label="compensation result",
    )
    plt.legend()
    # plt.show()


def test_digital_deconvolution_FIR_example_figure_6(
    simulated_measurement_input_and_output, invLSFIR_unc_filter_fit, fir_unc_filter
):
    plt.figure(figsize=(16, 8))
    plt.plot(
        simulated_measurement_input_and_output["time"] * 1e3,
        simulated_measurement_input_and_output["x"],
        label="input signal",
    )
    plt.plot(
        simulated_measurement_input_and_output["time"] * 1e3,
        simulated_measurement_input_and_output["yn"],
        label="output signal",
    )
    plt.plot(
        simulated_measurement_input_and_output["time"] * 1e3,
        fir_unc_filter["xhat"],
        label="estimate of input",
    )
    plt.legend(fontsize=20)
    plt.xlabel("time / ms", fontsize=22)
    plt.ylabel("signal amplitude / au", fontsize=22)
    plt.tick_params(which="both", labelsize=16)
    plt.xlim(1.9, 2.4)
    plt.ylim(-1, 1)


def test_digital_deconvolution_FIR_example_figure_7(
    simulated_measurement_input_and_output,
    fir_unc_filter,
):
    plt.figure(figsize=(16, 10))
    plt.plot(
        simulated_measurement_input_and_output["time"] * 1e3,
        fir_unc_filter["Uxhat"],
    )
    plt.xlabel("time / ms")
    plt.ylabel("signal uncertainty / au")
    plt.subplots_adjust(left=0.15, right=0.95)
    plt.tick_params(which="both", labelsize=16)
    plt.xlim(1.9, 2.4)


@given(hypothesis_dimension(min_value=2, max_value=12))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
    ],
)
@pytest.mark.slow
def test_compare_invLSFIR_unc_to_invLSFIR(
    monte_carlo, frequencies, sampling_frequency, filter_order
):
    bF_unc, _ = invLSFIR_unc(
        H=monte_carlo["H"],
        UH=np.zeros_like(monte_carlo["UH"]),
        N=filter_order,
        tau=filter_order // 2,
        f=frequencies,
        Fs=sampling_frequency,
    )
    bF = invLSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        tau=filter_order // 2,
        f=frequencies,
        Fs=sampling_frequency,
    )
    assert_allclose(
        bF_unc,
        bF,
    )


@given(hypothesis_dimension(min_value=2, max_value=12))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
    ],
)
def test_usual_call_LSFIR(monte_carlo, frequencies, sampling_frequency, filter_order):
    LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        tau=filter_order // 2,
        f=frequencies,
        Fs=sampling_frequency,
    )


@given(hypothesis_dimension(min_value=4, max_value=8))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
    ],
)
@pytest.mark.slow
def test_compare_invLSFIR_unc_to_invLSFIR_uncMC(
    monte_carlo, frequencies, sampling_frequency, filter_order
):
    b, ub = invLSFIR_unc(
        H=monte_carlo["H"],
        UH=monte_carlo["UH"],
        N=filter_order,
        tau=filter_order // 2,
        f=frequencies,
        Fs=sampling_frequency,
    )
    b_mc, ub_mc = invLSFIR_uncMC(
        H=monte_carlo["H"],
        UH=monte_carlo["UH"],
        N=filter_order,
        tau=filter_order // 2,
        f=frequencies,
        Fs=sampling_frequency,
    )
    assert_allclose(b, b_mc, rtol=4e-2)
    assert_allclose(ub, ub_mc, rtol=6e-1)
