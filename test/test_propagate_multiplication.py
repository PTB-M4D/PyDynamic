"""Test PyDynamic.uncertainty.propagate_multiplication"""

from typing import Callable

import numpy as np
import scipy.linalg as scl
import pytest
from hypothesis import given
from hypothesis.strategies import composite
from numpy.testing import assert_allclose

from PyDynamic.misc.tools import complex_2_real_imag as c2ri
from PyDynamic.misc.tools import real_imag_2_complex as ri2c
from PyDynamic.uncertainty.propagate_multiplication import (
    hadamar_product,
    window_application,
)
from .test_propagate_convolution import x_and_Ux

from .conftest import hypothesis_dimension


def hadamar_product_slow_reference(x1, U1, x2, U2, real_valued=False):
    """
    reference implementation to compare against in tests

    constructs the full sensitivity matrix (which is sparse), hence not efficient
    """

    #
    if real_valued:
        N = len(x1)

        prod = x1 * x2
        cov_prod = np.zeros((N, N))
        C1 = np.diag(x2)
        C2 = np.diag(x1)

    else:
        prod = c2ri(ri2c(x1) * ri2c(x2))
        N = len(x1) // 2
        cov_prod = np.zeros((2 * N, 2 * N))

        C1 = np.block(
            [
                [np.diag(x2[:N]), np.diag(-x2[N:])],
                [np.diag(x2[N:]), np.diag(x2[:N])],
            ]
        )

        C2 = np.block(
            [
                [np.diag(x1[:N]), np.diag(-x1[N:])],
                [np.diag(x1[N:]), np.diag(x1[:N])],
            ]
        )

    # actual calculation
    if isinstance(U1, np.ndarray):
        cov_prod += C1 @ U1 @ C1.T
    if isinstance(U2, np.ndarray):
        cov_prod += C2 @ U2 @ C2.T
    return prod, cov_prod


def window_application_slow_reference(A, W, cov_A=None, cov_W=None):
    """
    A \in R^2N uses PyDynamic real-imag representation of a complex vector \in C^N
    A = [A_re, A_im]

    W \in R^N is real-valued window

    R is result in real-imag representation, element-wise application of window (separately for real and imag values)
    R = [A_re * W, A_im * W]
    """
    N = len(W)
    R = A * np.r_[W, W]
    cov_R = np.zeros((2 * N, 2 * N))

    if isinstance(cov_A, np.ndarray):
        # this results from applying GUM
        CA = scl.block_diag(np.diag(W), np.diag(W))
        cov_R += CA @ cov_A @ CA.T

    if isinstance(cov_W, np.ndarray):
        # this results from applying GUM
        CW = np.vstack([np.diag(A[:N]), np.diag(A[N:])])
        cov_R += CW @ cov_W @ CW.T

    return R, cov_R


@composite
def two_x_and_Ux_of_same_length(draw: Callable, reduced_set=False):
    dim = 2 * draw(hypothesis_dimension(min_value=4, max_value=6))
    fixed_length_x_and_Ux = x_and_Ux(reduced_set=reduced_set, given_dim=dim)
    return draw(fixed_length_x_and_Ux), draw(fixed_length_x_and_Ux)


@composite
def complex_x_and_Ux_and_real_window(draw: Callable, reduced_set=False):
    real_dim = draw(hypothesis_dimension(min_value=4, max_value=6))
    complex_x_and_Ux = x_and_Ux(reduced_set=reduced_set, given_dim=2 * real_dim)
    real_window = x_and_Ux(reduced_set=reduced_set, given_dim=real_dim)
    return draw(complex_x_and_Ux), draw(real_window)


@given(two_x_and_Ux_of_same_length(reduced_set=True))
def test_hadamar(inputs):
    input_1, input_2 = inputs

    # calculate the hadamar product of input1 and input2
    y, Uy = hadamar_product(*input_1, *input_2)
    y_slow, Uy_slow = hadamar_product_slow_reference(*input_1, *input_2)

    # compare result sizes
    assert len(y) == len(Uy)
    assert len(y) == len(y_slow)
    assert len(y_slow) == len(Uy_slow)

    # compare results
    assert_allclose(y, y_slow, atol=1e2 * np.finfo(np.float64).eps)
    assert_allclose(Uy, Uy_slow, atol=1e2 * np.finfo(np.float64).eps)


@given(two_x_and_Ux_of_same_length(reduced_set=True))
def test_hadamar_real_valued(inputs):
    input_1, input_2 = inputs

    # calculate the convolution of x1 and x2
    y, Uy = hadamar_product(*input_1, *input_2, real_valued=True)
    y_slow, Uy_slow = hadamar_product_slow_reference(
        *input_1, *input_2, real_valued=True
    )

    # compare results
    assert len(y) == len(Uy)
    assert len(y) == len(y_slow)
    assert len(y_slow) == len(Uy_slow)

    assert_allclose(y, y_slow, atol=1e2 * np.finfo(np.float64).eps)
    assert_allclose(Uy, Uy_slow, atol=1e2 * np.finfo(np.float64).eps)


@given(two_x_and_Ux_of_same_length())
@pytest.mark.slow
def test_hadamar_common_call(inputs):
    input_1, input_2 = inputs

    # check common execution of convolve_unc
    assert hadamar_product(*input_1, *input_2)


def test_hadamar_known_result():
    # test against a result calculated by hand (on paper)

    x1 = np.array([1, 2, 3, 4], dtype=np.double)
    U1 = scl.toeplitz(np.flip(x1))
    x2 = x1 + 4
    U2 = U1 - 2

    r_known = np.array([-16, -20, 22, 40], dtype=np.double)
    Ur_known = np.array(
        [
            [176, 104, -48, -160],
            [104, 248, 56, -56],
            [-48, 56, 456, 432],
            [-160, -56, 432, 632],
        ],
        dtype=np.double,
    )

    r, Ur = hadamar_product(x1=x1, U1=U1, x2=x2, U2=U2)
    # check common execution of convolve_unc
    assert_allclose(r, r_known)
    assert_allclose(Ur, Ur_known)


def test_hadamar_known_result_real():
    # test against a result calculated by hand (on paper)

    x1 = np.array([1, 2, 3, 4], dtype=np.double)
    U1 = scl.toeplitz(np.flip(x1))
    x2 = x1 + 4
    U2 = U1 - 2

    r_known = np.array([5, 12, 21, 32], dtype=np.double)
    Ur_known = np.array(
        [
            [102, 92, 70, 36],
            [92, 152, 132, 96],
            [70, 132, 214, 180],
            [36, 96, 180, 288],
        ],
        dtype=np.double,
    )

    r, Ur = hadamar_product(x1=x1, U1=U1, x2=x2, U2=U2, real_valued=True)
    # check common execution of convolve_unc
    assert_allclose(r, r_known)
    assert_allclose(Ur, Ur_known)


@given(complex_x_and_Ux_and_real_window(reduced_set=True))
def test_window_application(inputs):
    x, Ux = inputs[0]
    w, Uw = inputs[1]

    # calculate the hadamar product of input1 and input2
    y, Uy = window_application(x, w, Ux, Uw)
    y_alt, Uy_alt = window_application_slow_reference(x, w, Ux, Uw)

    # compare result sizes
    assert len(y) == len(Uy)
    assert len(y) == len(y_alt)
    assert len(y_alt) == len(Uy_alt)

    # compare results
    assert_allclose(y, y_alt, atol=1e2 * np.finfo(np.float64).eps)
    assert_allclose(Uy, Uy_alt, atol=1e2 * np.finfo(np.float64).eps)


@given(complex_x_and_Ux_and_real_window())
@pytest.mark.slow
def test_window_application_common_call(inputs):
    x, Ux = inputs[0]
    w, Uw = inputs[1]

    # calculate the hadamar product of input1 and input2
    assert window_application(x, w, Ux, Uw)
