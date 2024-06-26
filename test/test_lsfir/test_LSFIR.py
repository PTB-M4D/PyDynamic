import os
import pathlib
from typing import cast

import hypothesis.strategies as hst
import numpy as np
import pytest
import scipy.signal as dsp
from hypothesis import given, HealthCheck, settings
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose

# noinspection PyProtectedMember
from PyDynamic import FIRuncFilter, kaiser_lowpass
from PyDynamic.model_estimation.fit_filter import (
    invLSFIR,
    LSFIR,
)
from .conftest import weights
from ..conftest import (
    hypothesis_dimension,
    custom_atol,
)


@pytest.fixture(scope="module")
def simulated_measurement_input_and_output(
    sampling_freq, digital_filter, random_number_generator
):
    Ts = 1 / sampling_freq
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
        * np.exp(-((time - t0) ** 2) / (2 * sigma**2))
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
        atol=custom_atol,
    )
    y = dsp.lfilter(digital_filter["b"], digital_filter["a"], x)
    noise = 1e-3
    yn = y + random_number_generator.standard_normal(np.size(y)) * noise

    return {"time": time, "x": x, "yn": yn, "noise": noise}


@pytest.fixture(scope="module")
def LSFIR_filter_fit(monte_carlo, freqs, sampling_freq):
    N = 12
    tau = N // 2
    bF, UbF = LSFIR(
        monte_carlo["H"], N, freqs, sampling_freq, tau, inv=True, UH=monte_carlo["UH"]
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
        atol=custom_atol,
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
        atol=custom_atol,
    )
    return {"bF": bF, "UbF": UbF, "N": N, "tau": tau}


@pytest.fixture(scope="module")
def fir_low_pass(measurement_system, sampling_freq):
    fcut = measurement_system["f0"] + 10e3
    low_order = 100
    blow, lshift = kaiser_lowpass(low_order, fcut, sampling_freq)
    assert_allclose(
        blow,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_blow.npz",
            ),
        )["blow"],
        atol=custom_atol,
    )
    return {"blow": blow, "lshift": lshift}


@pytest.fixture(scope="module")
def shift(simulated_measurement_input_and_output, LSFIR_filter_fit, fir_low_pass):
    shift = (len(LSFIR_filter_fit["bF"]) - 1) // 2 + fir_low_pass["lshift"]
    assert_allclose(
        shift,
        np.load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "reference_arrays",
                "test_LSFIR_shift.npz",
            ),
        )["shift"],
        atol=custom_atol,
    )
    return shift


@pytest.fixture(scope="module")
def fir_unc_filter(
    shift,
    simulated_measurement_input_and_output,
    LSFIR_filter_fit,
    fir_low_pass,
):
    xhat, Uxhat = FIRuncFilter(
        simulated_measurement_input_and_output["yn"],
        simulated_measurement_input_and_output["noise"],
        LSFIR_filter_fit["bF"],
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
    freqs, complex_freq_resp, monte_carlo
):
    plt.figure(figsize=(16, 8))
    plt.errorbar(
        freqs * 1e-3,
        np.abs(complex_freq_resp),
        monte_carlo["uAbs"],
        fmt=".",
    )
    plt.title("measured amplitude spectrum with associated uncertainties")
    plt.xlim(0, 50)
    plt.xlabel("frequency / kHz", fontsize=20)
    plt.ylabel("magnitude / au", fontsize=20)
    # plt.show()


def test_digital_deconvolution_FIR_example_figure_2(
    freqs, complex_freq_resp, monte_carlo
):
    plt.figure(figsize=(16, 8))
    plt.errorbar(
        freqs * 1e-3,
        np.angle(complex_freq_resp),
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
    LSFIR_filter_fit, monte_carlo, freqs, sampling_freq
):
    plt.figure(figsize=(16, 8))
    plt.errorbar(
        range(len(LSFIR_filter_fit["bF"])),
        LSFIR_filter_fit["bF"],
        np.sqrt(np.diag(LSFIR_filter_fit["UbF"])),
        fmt="o",
    )
    plt.xlabel("FIR coefficient index", fontsize=20)
    plt.ylabel("FIR coefficient value", fontsize=20)
    # plt.show()


def test_digital_deconvolution_FIR_example_figure_5(
    LSFIR_filter_fit, freqs, sampling_freq, complex_freq_resp, fir_low_pass
):
    plt.figure(figsize=(16, 10))
    HbF = (
        dsp.freqz(
            LSFIR_filter_fit["bF"],
            1,
            2 * np.pi * freqs / sampling_freq,
        )[1]
        * dsp.freqz(fir_low_pass["blow"], 1, 2 * np.pi * freqs / sampling_freq)[1]
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
        atol=custom_atol,
    )
    plt.semilogy(
        freqs * 1e-3,
        np.abs(complex_freq_resp),
        label="measured frequency response",
    )
    plt.semilogy(freqs * 1e-3, np.abs(HbF), label="inverse filter")
    plt.semilogy(
        freqs * 1e-3,
        np.abs(complex_freq_resp * HbF),
        label="compensation result",
    )
    plt.legend()
    # plt.show()


def test_digital_deconvolution_FIR_example_figure_6(
    simulated_measurement_input_and_output, fir_unc_filter
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


@given(hypothesis_dimension(min_value=4, max_value=8))
@settings(deadline=None)
def test_LSFIR_with_too_short_H(monte_carlo, freqs, sampling_freq, filter_order):
    too_short_H = monte_carlo["H"][1:]
    with pytest.raises(
        ValueError,
        match=r"LSFIR: vector of complex frequency responses is expected to "
        r"contain [0-9]+ elements, corresponding to the number of frequencies.*",
    ):
        LSFIR(
            H=too_short_H,
            N=filter_order,
            f=freqs,
            Fs=sampling_freq,
            tau=filter_order // 2,
            inv=True,
            UH=monte_carlo["UH"],
            mc_runs=2,
        )


@given(hypothesis_dimension(min_value=4, max_value=8))
@settings(deadline=None)
def test_LSFIR_with_complex_but_too_short_H(
    complex_H_with_UH, freqs, sampling_freq, filter_order
):
    complex_h_but_too_short = complex_H_with_UH["H"][1:]
    with pytest.raises(
        ValueError,
        match=r"LSFIR: vector of complex frequency responses is expected to "
        r"contain [0-9]+ elements, corresponding to the number of frequencies.*",
    ):
        LSFIR(
            H=complex_h_but_too_short,
            N=filter_order,
            f=freqs,
            Fs=sampling_freq,
            tau=filter_order // 2,
            inv=True,
            UH=complex_H_with_UH["UH"],
            mc_runs=2,
        )


@given(hypothesis_dimension(min_value=4, max_value=8))
@settings(deadline=None)
def test_LSFIR_with_too_short_f(monte_carlo, freqs, sampling_freq, filter_order):
    too_short_f = freqs[1:]
    with pytest.raises(
        ValueError,
        match=r"LSFIR: vector of complex frequency responses is expected to "
        r"contain [0-9]+ elements, corresponding to the number of frequencies.*",
    ):
        LSFIR(
            H=monte_carlo["H"],
            N=filter_order,
            f=too_short_f,
            Fs=sampling_freq,
            tau=filter_order // 2,
            inv=True,
            UH=monte_carlo["UH"],
            mc_runs=2,
        )


@given(hypothesis_dimension(min_value=4, max_value=8))
@settings(deadline=None)
def test_LSFIR_with_too_short_UH(monte_carlo, freqs, sampling_freq, filter_order):
    too_few_rows_UH = monte_carlo["UH"][1:]
    with pytest.raises(
        ValueError,
        match=r"LSFIR: number of rows of uncertainties and number of "
        r"elements of values are expected to match\..*",
    ):
        LSFIR(
            H=monte_carlo["H"],
            N=filter_order,
            f=freqs,
            Fs=sampling_freq,
            tau=filter_order // 2,
            inv=True,
            UH=too_few_rows_UH,
            mc_runs=2,
        )


@given(hypothesis_dimension(min_value=4, max_value=8))
@settings(deadline=None)
def test_LSFIR_with_nonsquare_UH(monte_carlo, freqs, sampling_freq, filter_order):
    too_few_columns_UH = monte_carlo["UH"][:, 1:]
    with pytest.raises(
        ValueError,
        match=r"LSFIR: uncertainties are expected to be "
        r"provided in a square matrix shape.*",
    ):
        LSFIR(
            H=monte_carlo["H"],
            N=filter_order,
            f=freqs,
            Fs=sampling_freq,
            tau=filter_order // 2,
            inv=True,
            UH=too_few_columns_UH,
            mc_runs=2,
        )


@given(hypothesis_dimension(min_value=4, max_value=8))
@settings(deadline=None)
def test_LSFIR_with_wrong_type_UH(monte_carlo, freqs, sampling_freq, filter_order):
    uh_list = monte_carlo["UH"].tolist()
    with pytest.raises(
        TypeError,
        match=r"LSFIR: if uncertainties are provided, "
        r"they are expected to be of type np\.ndarray.*",
    ):
        LSFIR(
            H=monte_carlo["H"],
            N=filter_order,
            f=freqs,
            Fs=sampling_freq,
            tau=filter_order // 2,
            inv=True,
            UH=cast(np.ndarray, uh_list),
            mc_runs=2,
        )


@given(hypothesis_dimension(min_value=4, max_value=8))
@settings(deadline=None)
def test_LSFIR_with_wrong_type_weights(monte_carlo, freqs, sampling_freq, filter_order):
    weight_list = [1] * 2 * len(freqs)
    with pytest.raises(
        TypeError, match=r"LSFIR: User-defined weighting has wrong type.*"
    ):
        LSFIR(
            H=monte_carlo["H"],
            N=filter_order,
            f=freqs,
            Fs=sampling_freq,
            tau=filter_order // 2,
            weights=cast(np.ndarray, weight_list),
            inv=True,
            UH=monte_carlo["UH"],
            mc_runs=2,
        )


@given(weights(guarantee_vector=True), hypothesis_dimension(min_value=4, max_value=8))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
    ],
)
@pytest.mark.slow
def test_LSFIR_with_wrong_len_weights(
    monte_carlo, freqs, sampling_freq, weight_vector, filter_order
):
    weight_vector = weight_vector[1:]
    with pytest.raises(
        ValueError,
        match=r"LSFIR: User-defined weighting has wrong dimension.*",
    ):
        LSFIR(
            H=monte_carlo["H"],
            N=filter_order,
            f=freqs,
            Fs=sampling_freq,
            tau=filter_order // 2,
            weights=weight_vector,
            inv=True,
            UH=monte_carlo["UH"],
            mc_runs=2,
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
def test_not_implemented_LSFIR(monte_carlo, freqs, sampling_freq, filter_order):
    expected_error_msg_regex = (
        r"LSFIR: The least-squares fitting of a digital FIR filter "
        r".*truncated singular-value decomposition and linear matrix propagation.*is "
        r"not yet implemented.*"
    )
    with pytest.raises(
        NotImplementedError,
        match=expected_error_msg_regex,
    ):
        LSFIR(
            H=monte_carlo["H"],
            UH=monte_carlo["UH"],
            N=filter_order,
            tau=filter_order // 2,
            f=freqs,
            Fs=sampling_freq,
            inv=False,
        )
    with pytest.raises(
        NotImplementedError,
        match=expected_error_msg_regex,
    ):
        LSFIR(
            H=monte_carlo["H"],
            UH=monte_carlo["UH"],
            N=filter_order,
            tau=filter_order // 2,
            f=freqs,
            Fs=sampling_freq,
            inv=False,
            trunc_svd_tol=0.0,
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
def test_missing_mc_uncertainties_LSFIR(
    monte_carlo, freqs, sampling_freq, filter_order
):
    with pytest.raises(
        ValueError,
        match=r"LSFIR: The least-squares fitting of a digital FIR filter "
        r".*Monte Carlo.*requires that uncertainties are provided via input "
        r"parameter UH.*",
    ):
        LSFIR(
            H=monte_carlo["H"],
            N=filter_order,
            tau=filter_order // 2,
            f=freqs,
            Fs=sampling_freq,
            inv=False,
            mc_runs=1,
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
def test_missing_svd_uncertainties_LSFIR(
    monte_carlo, freqs, sampling_freq, filter_order
):
    with pytest.raises(
        ValueError,
        match=r"LSFIR: The least-squares fitting of a digital FIR filter "
        r".*singular-value decomposition and linear matrix propagation.*requires that "
        r"uncertainties are provided via input parameter UH.*",
    ):
        LSFIR(
            H=monte_carlo["H"],
            N=filter_order,
            tau=filter_order // 2,
            f=freqs,
            Fs=sampling_freq,
            inv=True,
            trunc_svd_tol=0.0,
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
def test_both_propagation_methods_simultaneously_requested_uncertainties_LSFIR(
    monte_carlo, freqs, sampling_freq, filter_order
):
    with pytest.raises(
        ValueError,
        match=r"LSFIR: Only one of mc_runs and trunc_svd_tol can be " r"provided but.*",
    ):
        LSFIR(
            H=monte_carlo["H"],
            UH=monte_carlo["UH"],
            N=filter_order,
            tau=filter_order // 2,
            f=freqs,
            Fs=sampling_freq,
            inv=False,
            mc_runs=2,
            trunc_svd_tol=0.0,
        )


@given(hypothesis_dimension(min_value=4, max_value=8), hst.booleans())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
    ],
)
@pytest.mark.slow
def test_too_small_number_of_monte_carlo_runs_LSFIR(
    monte_carlo, freqs, sampling_freq, filter_order, inv
):
    with pytest.raises(
        ValueError,
        match=r"LSFIR: Number of Monte Carlo runs is expected to be greater "
        r"than 1.*",
    ):
        LSFIR(
            H=monte_carlo["H"],
            UH=monte_carlo["UH"],
            N=filter_order,
            tau=filter_order // 2,
            f=freqs,
            Fs=sampling_freq,
            inv=inv,
            mc_runs=1,
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
def test_compare_invLSFIR_to_LSFIR(monte_carlo, freqs, sampling_freq, filter_order):
    b_fir_inv_lsfir = invLSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        tau=filter_order // 2,
        f=freqs,
        Fs=sampling_freq,
    )
    b_fir = LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        f=freqs,
        Fs=sampling_freq,
        tau=filter_order // 2,
        inv=True,
    )[0]
    assert_allclose(b_fir, b_fir_inv_lsfir, atol=custom_atol)
