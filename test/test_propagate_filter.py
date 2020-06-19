"""
Perform test for uncertainty.propagate_filter
"""

import numpy as np
import pytest
import scipy

from PyDynamic.misc.filterstuff import kaiser_lowpass
from PyDynamic.misc.noise import power_law_acf, power_law_noise, white_gaussian
from PyDynamic.misc.testsignals import rect
from PyDynamic.misc.tools import make_semiposdef
from PyDynamic.uncertainty.propagate_filter import (
    FIRuncFilter,
    IIRuncFilter,
    _tf2ss,
    _get_derivative_A,
)

# parameters of simulated measurement
Fs = 100e3  # sampling frequency (in Hz)
Ts = 1 / Fs  # sampling interval length (in s)

# nominal system parameters
fcut = 20e3  # low-pass filter cut-off frequency (6 dB)
L = 100  # filter order
b1 = kaiser_lowpass(L, fcut, Fs)[0]
b2 = kaiser_lowpass(L - 20, fcut, Fs)[0]

# uncertain knowledge: cutoff between 19.5kHz and 20.5kHz
runs = 1000
FC = fcut + (2 * np.random.rand(runs) - 1) * 0.5e3

B = np.zeros((runs, L + 1))
for k in range(runs):  # Monte Carlo for filter coefficients of low-pass filter
    B[k, :] = kaiser_lowpass(L, FC[k], Fs)[0]

Ub = make_semiposdef(np.cov(B, rowvar=0))  # covariance matrix of MC result

# simulate input and output signals
nTime = 500
time = np.arange(nTime) * Ts  # time values

# different cases
sigma_noise = 1e-2  # 1e-5


def test_FIRuncFilter_float():

    # input signal + run methods
    x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=sigma_noise)

    # apply uncertain FIR filter (GUM formula)
    for blow in [None, b2]:
        y, Uy = FIRuncFilter(x, sigma_noise, b1, Ub, blow=blow, kind="float")
        assert len(y) == len(x)
        assert len(Uy) == len(x)


def test_FIRuncFilter_corr():

    # get an instance of noise, the covariance and the covariance-matrix with the specified color
    color = "white"
    noise = power_law_noise(N=nTime, color_value=color, std=sigma_noise)
    Ux = power_law_acf(nTime, color_value=color, std=sigma_noise)

    # input signal
    x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=noise)

    # apply uncertain FIR filter (GUM formula)
    for blow in [None, b2]:
        y, Uy = FIRuncFilter(x, Ux, b1, Ub, blow=blow, kind="corr")
        assert len(y) == len(x)
        assert len(Uy) == len(x)


def test_FIRuncFilter_diag():
    sigma_diag = sigma_noise * (
        1 + np.heaviside(np.arange(len(time)) - len(time) // 2, 0)
    )  # std doubles after half of the time
    noise = sigma_diag * white_gaussian(len(time))

    # input signal + run methods
    x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=noise)

    # apply uncertain FIR filter (GUM formula)
    for blow in [None, b2]:
        y, Uy = FIRuncFilter(x, sigma_diag, b1, Ub, blow=blow, kind="diag")
        assert len(y) == len(x)
        assert len(Uy) == len(x)


@pytest.fixture(scope="module")
def iir_filter():
    b = np.array([0.01967691, -0.01714282, 0.03329653, -0.01714282, 0.01967691])
    a = np.array([1.0, -3.03302405, 3.81183153, -2.29112937, 0.5553678])
    Uab = np.diag(np.zeros((len(a) + len(b) - 1)))  # uncertainty of IIR-parameters
    Uab[2, 2] = 0.000001  # only a2 is uncertain

    return {"b": b, "a": a, "Uab": Uab}


@pytest.fixture(scope="module")
def fir_filter():
    b = np.array([0.01967691, -0.01714282, 0.03329653, -0.01714282, 0.01967691])
    a = np.array([1.0,])
    Uab = np.diag(
        np.zeros((len(a) + len(b) - 1))
    )  # fully certain filter-parameters, otherwise FIR and IIR do not match! (see docstring of IIRuncFilter)

    return {"b": b, "a": a, "Uab": Uab}


@pytest.fixture(scope="module")
def input_signal():

    # simulate input and output signals
    Fs = 100e3  # sampling frequency (in Hz)
    Ts = 1 / Fs  # sampling interval length (in s)
    nx = 500
    time = np.arange(nx) * Ts  # time values

    # input signal + run methods
    sigma_noise = 1e-2  # std for input signal
    x = rect(time, 100 * Ts, 250 * Ts, 1.0, noise=sigma_noise)  # generate input signal
    Ux = sigma_noise * np.ones_like(x)  # uncertainty of input signal

    return {"x": x, "Ux": Ux}


@pytest.fixture(scope="module")
def run_IIRuncFilter_all_at_once(iir_filter, input_signal):
    y, Uy, _ = IIRuncFilter(**input_signal, **iir_filter, kind="diag" )
    return y, Uy

@pytest.fixture(scope="module")
def run_IIRuncFilter_in_chunks(iir_filter, input_signal):

    # slice x into smaller chunks and process them in batches
    # this tests the internal state options
    y_list = []
    Uy_list = []
    state = None
    for x_batch, Ux_batch in zip(
        np.array_split(input_signal["x"], 200), np.array_split(input_signal["Ux"], 200)
    ):
        yi, Uyi, state = IIRuncFilter(
            x_batch, Ux_batch, **iir_filter, kind="diag", state=state,
        )
        y_list.append(yi)
        Uy_list.append(Uyi)
    y = np.concatenate(y_list, axis=0)
    Uy = np.concatenate(Uy_list, axis=0)

    return y, Uy


def test_IIRuncFilter_shape_all_at_once(input_signal, run_IIRuncFilter_all_at_once):

    y, Uy = run_IIRuncFilter_all_at_once

    # compare lengths
    assert len(y) == len(input_signal["x"])
    assert len(Uy) == len(input_signal["Ux"])


def test_IIRuncFilter_shape_in_chunks(
    input_signal, run_IIRuncFilter_in_chunks
):
    y, Uy = run_IIRuncFilter_in_chunks

    # compare lengths
    assert len(y) == len(input_signal["x"])
    assert len(Uy) == len(input_signal["Ux"])


def test_IIRuncFilter_identity_nonchunk_chunk(
    run_IIRuncFilter_all_at_once, run_IIRuncFilter_in_chunks
):

    y1, Uy1 = run_IIRuncFilter_all_at_once
    y2, Uy2 = run_IIRuncFilter_in_chunks

    # check if both ways of calling IIRuncFilter yield the same result
    assert np.allclose(y1, y2)
    assert np.allclose(Uy1, Uy2)

@pytest.mark.parametrize("kind", ["diag", "corr"])
def test_FIR_IIR_identity(kind, fir_filter, input_signal):

    # run signal through both implementations
    y_iir, Uy_iir, _ = IIRuncFilter(*input_signal.values(), **fir_filter, kind=kind)
    y_fir, Uy_fir = FIRuncFilter(
        *input_signal.values(),
        theta=fir_filter["b"],
        Utheta=fir_filter["Uab"],
        kind=kind,
    )

    assert np.allclose(y_fir, y_iir)
    assert np.allclose(Uy_fir, Uy_iir)


def test_tf2ss(iir_filter):
    """compare output of _tf2ss to (the very similar) scipy.signal.tf2ss"""
    b = iir_filter["b"]
    a = iir_filter["a"]
    A1, B1, C1, D1 = _tf2ss(b, a)
    A2, B2, C2, D2 = scipy.signal.tf2ss(b, a)

    assert np.allclose(A1, A2[::-1, ::-1])
    assert np.allclose(B1, B2[::-1, ::-1])
    assert np.allclose(C1, C2[::-1, ::-1])
    assert np.allclose(D1, D2[::-1, ::-1])


def test_get_derivative_A():
    """dA is sparse and only a specifc diagonal is of value -1.0"""
    p = 10
    dA = _get_derivative_A(p)
    index1 = np.arange(p)
    index2 = np.full(p, -1)
    index3 = index1[::-1]

    sliced_diagonal = np.full(p, -1.0)

    assert np.allclose(dA[index1, index2, index3], sliced_diagonal)
