"""This module assists in uncertainty propagation for multiplication tasks

The multiplication of signals is a common operation in signal and data
processing.

This module contains the following functions:

* :func:`hadamar_product`: Elementwise Multiplication of two signals
* :func:`window_application`: Application of a real-valued window to a complex signal
"""

__all__ = ["hadamar_product", "window_application"]

import numpy as np
from scipy.linalg import block_diag
from typing import Optional

from PyDynamic.uncertainty.propagate_convolution import _ensure_cov_matrix


def hadamar_product(
    x1: np.ndarray,
    U1: Optional[np.ndarray],
    x2: np.ndarray,
    U2: Optional[np.ndarray],
    real_valued: bool = False,
):
    """Hadamar product of two uncorrelated signals with uncertainty propagation

    This is also known as elementwise multiplication.

    By default, both input signals are assumed to represent a complex signal,
    where the real and imaginary part are concatenated into a single vector:
    [Re(x), Im(x)]

    Parameters
    ----------
    x1 : np.ndarray, (2N,) or (N,)
        first input signal
    U1 : np.ndarray, (2N, 2N), (N, N) or (2N,), (N,)
        - 1D-array: standard uncertainties associated with x1 (corresponding to uncorrelated entries of x1)
        - 2D-array: full 2D-covariance matrix associated with x1
        - None: corresponds to a fully certain signal x1, results in more efficient calculation (compared to using np.zeros(...))
    x2 : np.ndarray, (2N,) or (N,)
        second input signal, same length as x1
    U2 : np.ndarray, (2N, 2N), (N, N) or (2N,), (N,)
        - 2D-array: full 2D-covariance matrix associated with x2
        - 1D-array: standard uncertainties associated with x2 (corresponding to uncorrelated entries of x2)
        - None: corresponds to a fully certain signal x2, results in more efficient calculation (compared to using np.zeros(...))
    real_valued : bool, optional
        By default, both input signals are assumed to represent a complex signal,
        where the real and imaginary part are concatenated into a single vector
        [Re(x), Im(x)].
        Alternatively, if both represent purely real signals, performance gains can be
        achieved by enabling this switch.

    Returns
    -------
    prod : np.ndarray, (2N,) or (N,)
        multiplied output signal
    Uprod : np.ndarray, (2N, 2N) or (N, N)
        full 2D-covariance matrix of output prod

    References
    ----------
    .. seealso:: :func:`np.multiply`
    """

    # check input lengths
    assert len(x1) == len(x2)

    # deal with std-unc-only case
    U1, U2 = _ensure_cov_matrix(U1), _ensure_cov_matrix(U2)

    # simplified calculations for real-valued signals
    if real_valued:
        N = len(x1)

        prod = x1 * x2

        cov_prod = np.zeros((N, N))
        if isinstance(U1, np.ndarray):
            cov_prod += np.atleast_2d(x2).T * U1 * np.atleast_2d(x2)
        if isinstance(U2, np.ndarray):
            cov_prod += np.atleast_2d(x1).T * U2 * np.atleast_2d(x1)

    else:
        N = len(x1) // 2

        # main calculations
        prod = np.empty_like(x1)
        prod[:N] = x1[:N] * x2[:N] - x1[N:] * x2[N:]
        prod[N:] = x1[N:] * x2[:N] + x1[:N] * x2[N:]

        # cov calculations
        cov_prod = np.zeros((2 * N, 2 * N))
        if isinstance(U1, np.ndarray):
            x2r = np.atleast_2d(x2[:N])
            x2i = np.atleast_2d(x2[N:])
            U1rr = U1[:N, :N]
            U1ri = U1[:N, N:]
            U1ir = U1[N:, :N]
            U1ii = U1[N:, N:]

            cov_prod[:N, :N] += (
                +x2r.T * U1rr * x2r
                - x2r.T * U1ri * x2i
                - x2i.T * U1ir * x2r
                + x2i.T * U1ii * x2i
            )
            cov_prod[:N, N:] += (
                +x2r.T * U1rr * x2i
                + x2r.T * U1ri * x2r
                - x2i.T * U1ir * x2i
                - x2i.T * U1ii * x2r
            )
            cov_prod[N:, :N] = cov_prod[:N, N:].T
            cov_prod[N:, N:] += (
                +x2i.T * U1rr * x2i
                + x2i.T * U1ri * x2r
                + x2r.T * U1ir * x2i
                + x2r.T * U1ii * x2r
            )
        if isinstance(U2, np.ndarray):
            x1r = np.atleast_2d(x1[:N])
            x1i = np.atleast_2d(x1[N:])
            U2rr = U2[:N, :N]
            U2ri = U2[:N, N:]
            U2ir = U2[N:, :N]
            U2ii = U2[N:, N:]

            cov_prod[:N, :N] += (
                +x1r.T * U2rr * x1r
                - x1r.T * U2ri * x1i
                - x1i.T * U2ir * x1r
                + x1i.T * U2ii * x1i
            )
            cov_prod[:N, N:] += (
                +x1r.T * U2rr * x1i
                + x1r.T * U2ri * x1r
                - x1i.T * U2ir * x1i
                - x1i.T * U2ii * x1r
            )
            cov_prod[N:, :N] = cov_prod[:N, N:].T
            cov_prod[N:, N:] += (
                +x1i.T * U2rr * x1i
                + x1i.T * U2ri * x1r
                + x1r.T * U2ir * x1i
                + x1r.T * U2ii * x1r
            )

    return prod, cov_prod


def window_application(
    A: np.ndarray,
    W: np.ndarray,
    cov_A: Optional[np.ndarray] = None,
    cov_W: Optional[np.ndarray] = None,
):
    """Application of a real window to a complex signal

    Parameters
    ----------
    A : np.ndarray, (2N,)
        signal the window will be applied to
    W : np.ndarray, (N,)
        window
    cov_A : np.ndarray, (2N,2N) or (2N,)
        - 2D-array: full 2D-covariance matrix associated with A
        - 1D-array: standard uncertainties associated with A (corresponding to uncorrelated entries of A)
        - None: corresponds to a fully certain signal A, results in more efficient calculation (compared to using np.zeros(...))
    cov_W : np.ndarray, (N, N) or (N,)
        - 2D-array: full 2D-covariance matrix associated with W
        - 1D-array: standard uncertainties associated with W (corresponding to uncorrelated entries of W)
        - None: corresponds to a fully certain signal W, results in more efficient calculation (compared to using np.zeros(...))

    Returns
    -------
    y : np.ndarray
        signal with window applied
    Uy : np.ndarray
        full 2D-covariance matrix of windowed signal

    References
    ----------
    .. seealso:: :func:`np.multiply`
    """

    # deal with std-unc-only case
    cov_A, cov_W = _ensure_cov_matrix(cov_A), _ensure_cov_matrix(cov_W)

    # map to variables of hadamar product
    x1 = A
    U1 = cov_A
    x2 = np.r_[W, np.zeros_like(W)]
    if isinstance(cov_W, np.ndarray):
        U2 = block_diag(cov_W, np.zeros_like(cov_W))
    else:
        U2 = None

    R, cov_R = hadamar_product(x1, U1, x2, U2)

    return R, cov_R
