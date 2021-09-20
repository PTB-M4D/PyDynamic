# -*- coding: utf-8 -*-

"""
The :mod:`PyDynamic.uncertainty.propagate_DFT` module implements methods for
the propagation of uncertainties in the application of the DFT, inverse DFT,
deconvolution and multiplication in the frequency domain, transformation from
amplitude and phase to real and imaginary parts and vice versa.

The corresponding scientific publications is
    S. Eichstädt und V. Wilkens
    GUM2DFT — a software tool for uncertainty evaluation of transient signals
    in the frequency domain. *Measurement Science and Technology*, 27(5),
    055001, 2016. [DOI: `10.1088/0957-0233/27/5/055001
    <http://dx.doi.org/10.1088/0957-0233/27/5/055001>`_]

This module contains the following functions:

* :func:`GUM_DFT`: Calculation of the DFT of the time domain signal x and
  propagation of the squared uncertainty Ux associated with the time domain
  sequence x to the real and imaginary parts of the DFT of x
* :func:`GUM_iDFT`: GUM propagation of the squared uncertainty UF associated with
  the DFT values F through the inverse DFT
* :func:`GUM_DFTfreq`: Return the Discrete Fourier Transform sample frequencies
* :func:`DFT_transferfunction`: Calculation of the transfer function H = Y/X in the
  frequency domain with X being the Fourier transform
  of the system's input signal and Y that of the output signal
* :func:`DFT_deconv`: Deconvolution in the frequency domain
* :func:`DFT_multiply`: Multiplication in the frequency domain
* :func:`AmpPhase2DFT`: Transformation from magnitude and phase to real and
  imaginary parts
* :func:`DFT2AmpPhase`: Transformation from real and imaginary parts to magnitude
  and phase
* :func:`AmpPhase2Time`: Transformation from amplitude and phase to time domain
* :func:`Time2AmpPhase`: Transformation from time domain to amplitude and phase
"""

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import sparse

__all__ = [
    "GUM_DFT",
    "GUM_iDFT",
    "GUM_DFTfreq",
    "DFT_transferfunction",
    "DFT_deconv",
    "DFT_multiply",
    "AmpPhase2DFT",
    "DFT2AmpPhase",
    "AmpPhase2Time",
    "Time2AmpPhase",
    "Time2AmpPhase_multi",
]

from PyDynamic.misc.tools import (
    is_2d_matrix,
    is_vector,
    number_of_rows_equals_vector_dim,
)


def GUM_DFT(
    x: np.ndarray,
    Ux: Union[np.ndarray, float],
    N: Optional[int] = None,
    window: Optional[np.ndarray] = None,
    CxCos: Optional[np.ndarray] = None,
    CxSin: Optional[np.ndarray] = None,
    returnC: Optional[bool] = False,
    mask: Optional[np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]],
    Tuple[
        np.ndarray,
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        Dict[str, np.ndarray],
    ],
]:
    """Calculation of the DFT with propagation of uncertainty

    Calculation of the DFT of the time domain signal x and propagation of
    the squared uncertainty Ux associated with the time domain sequence x to
    the real and imaginary parts of the DFT of x.

    Parameters
    ----------
    x : np.ndarray of shape (M,)
        vector of time domain signal values
    Ux : np.ndarray of shape (M,) or of shape (M,M) or float
        covariance matrix associated with x, or vector of squared standard
        uncertainties, or noise variance as float
    N : int, optional
        length of time domain signal for DFT; N>=len(x)
    window : np.ndarray of shape (M,), optional
        vector of the time domain window values
    CxCos : np.ndarray, optional
        cosine part of sensitivity matrix
    CxSin : np.ndarray, optional
        sine part of sensitivity matrix
    returnC : bool, optional
        if True, return sensitivity matrix blocks, if False (default) do not return them
    mask: ndarray of dtype bool, optional
        calculate DFT values and uncertainties only at those frequencies
        where mask is True

    Returns
    -------
    F : np.ndarray
        vector of complex valued DFT values or of its real and imaginary parts
    UF : np.ndarray
        covariance matrix associated with real and imaginary part of F
    CxCos and CxSin : Dict
        Keys are "CxCos", "CxSin" and values the respective sensitivity matrix entries

    References
    ----------
        * Eichstädt and Wilkens [Eichst2016]_

    Raises
    ------
    ValueError
        If N < len(x)
    """
    L = 0
    # Apply the chosen window for the application of the FFT.
    if isinstance(window, np.ndarray):
        x, Ux = _apply_window(x, Ux, window)
    if isinstance(N, int):
        L = N - len(x)
        if L < 0:
            raise ValueError(
                "N needs to be greater or equal than the length of x, "
                f"but N = {N} and len(x) = {len(x)} were given."
            )
        x = np.r_[
            x.copy(),
            np.zeros(
                L,
            ),
        ]  # zero-padding
    N = len(x)
    if np.mod(N, 2) == 0:  # N is even
        M = N + 2
    else:
        M = N + 1

    if isinstance(mask, np.ndarray):
        F = np.fft.rfft(x)[mask]  # calculation of best estimate
        # In real, imag format in accordance with GUM S2
        F = np.r_[np.real(F), np.imag(F)]
        warnings.warn(
            "In a future release, because of issues with the current version, "
            "\nthe handling of masked DFT arrays will be changed to use numpy "
            "masked arrays.",
            DeprecationWarning,
        )
    else:
        F = np.fft.rfft(x)  # calculation of best estimate
        # In real, imag format in accordance with GUM S2
        F = np.r_[np.real(F), np.imag(F)]
        mask = np.ones(len(F) // 2, dtype=bool)
    Nm = 2 * np.sum(mask)

    # For simplified calculation of sensitivities
    beta = 2 * np.pi * np.arange(N - L) / N

    # sensitivity matrix wrt cosine part
    Cxkc = lambda k: np.cos(k * beta)[np.newaxis, :]
    # sensitivity matrix wrt sinus part
    Cxks = lambda k: -np.sin(k * beta)[np.newaxis, :]

    if isinstance(Ux, float):
        UF = np.zeros(Nm)
        km = 0
        for k in range(M // 2):  # Block cos/cos
            if mask[k]:
                UF[km] = np.sum(Ux * Cxkc(k) ** 2)
                km += 1
        km = 0
        for k in range(M // 2):  # Block sin/sin
            if mask[k]:
                UF[Nm // 2 + km] = np.sum(Ux * Cxks(k) ** 2)
                km += 1
    else:  # general method
        if len(Ux.shape) == 1:
            Ux = np.diag(Ux)
        if not isinstance(CxCos, np.ndarray):
            CxCos = np.zeros((Nm // 2, N - L))
            CxSin = np.zeros((Nm // 2, N - L))
            km = 0
            for k in range(M // 2):
                if mask[k]:
                    CxCos[km, :] = Cxkc(k)
                    CxSin[km, :] = Cxks(k)
                    km += 1
        UFCC = np.dot(CxCos, np.dot(Ux, CxCos.T))
        UFCS = np.dot(CxCos, np.dot(Ux, CxSin.T))
        UFSS = np.dot(CxSin, np.dot(Ux, CxSin.T))
        try:
            UF = np.vstack(
                (np.hstack((UFCC, UFCS)), np.hstack((UFCS.T, UFSS)))
            )  # type: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        except MemoryError:
            print("Could not put covariance matrix together due to memory constraints.")
            print(
                "Returning the three blocks (A,B,C) such that U = [[A,B],"
                "[B.T,C]] instead."
            )
            # Return blocks only because of lack of memory.
            UF = (UFCC, UFCS, UFSS)  # type:
            # Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

    if returnC:
        # Return sensitivities if requested.
        return F, UF, {"CxCos": CxCos, "CxSin": CxSin}
    else:
        return F, UF


def _apply_window(
    x: np.ndarray, Ux: Union[np.ndarray, float], window: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply time domain window to signal x of equal length and propagate uncertainties

    Parameters
    ----------
    x : np.ndarray of shape (N, )
        vector of time domain signal values
    Ux : np.ndarray of shape (N, N) or float
        covariance matrix associated with x or noise variance as float
    window : np.ndarray of shape (N, )
        vector of time domain window (same length as x)

    Returns
    -------
    xw, Uxw : np.ndarray of shape (N, ) and (N, N)
        transformed signal and associated uncertainties

    Raises
    ------
    AssertionError
        If the dimensions of x, Ux and window do not match or Ux is not square.
    """
    assert len(x) == len(window)
    if not isinstance(Ux, float):
        assert Ux.shape[0] == Ux.shape[1] and Ux.shape[0] == len(x)
    xw = x.copy() * window
    if isinstance(Ux, float):
        Uxw = Ux * window ** 2
    else:
        Uxw = _prod(window, _prod(Ux, window))
    return xw, Uxw


def _prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""Calculate the product of a matrix with a diagonal matrix of a vector

    Calculate the product that corresponds to :math:`diag(a) \cdot b` if :math:`a`
    is the vector or :math:`a \cdot diag(b)` else.

    Parameters
    ----------
    a, b : np.ndarray of shape (N,M) and np.ndarray of shape (N,) on of shape  (M,)
        one is a vector from which the diagonal matrix is build and the other a
        matrix where the order does not matter

    Returns
    -------
    np.ndarray
        The product of the matrices

    Raises
    ------
    AssertionError
        If the dimensions of a and b do not match
    """
    assert _check_matrix_vector_dimension_match(a=a, b=b), (
        "_prod: Wrong dimension of inputs. Expected 'b' to be matrix with "
        "the same number of columns as the vector 'a' or 'a' to be a matrix with "
        "the same number of rows as the vector 'b'. The shape of 'a' is "
        f"{a.shape} and of 'b' is {b.shape}."
    )
    if is_vector(ndarray=a):
        return _multiply_diagonal_matrix_from_vector_with_matrix_from_left(
            matrix=b, vector=a
        )
    else:
        return _multiply_diagonal_matrix_from_vector_with_matrix_from_right(
            matrix=a, vector=b
        )


def _check_matrix_vector_dimension_match(a: np.ndarray, b: np.ndarray) -> bool:
    return (
                   is_vector(a)
                   and is_2d_matrix(b)
                   and number_of_rows_equals_vector_dim(vector=a, matrix=b)
    ) or (
                   is_vector(b)
                   and is_2d_matrix(a)
                   and _number_of_cols_equals_vector_dim(vector=b, matrix=a)
    )


def _number_of_cols_equals_vector_dim(matrix: np.ndarray, vector: np.ndarray) -> bool:
    return len(vector) == matrix.shape[1]


def _multiply_diagonal_matrix_from_vector_with_matrix_from_right(
    matrix: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    return matrix @ np.diag(v=vector)


def _multiply_diagonal_matrix_from_vector_with_matrix_from_left(
    matrix: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    return np.diag(v=vector) @ matrix


def GUM_iDFT(
    F: np.ndarray,
    UF: np.ndarray,
    Nx: Optional[int] = None,
    Cc: Optional[np.ndarray] = None,
    Cs: Optional[np.ndarray] = None,
    returnC: Optional[bool] = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[
        np.ndarray,
        np.ndarray,
        Dict[str, np.ndarray],
    ],
]:
    """Propagation of squared uncertainties UF associated with the DFT values F

    GUM propagation of the squared uncertainties UF associated with the DFT
    values F through the inverse DFT.

    The matrix UF is assumed to be for real and imaginary part with blocks:
    UF = [[u(R,R), u(R,I)],[u(I,R),u(I,I)]]
    and real and imaginary part obtained from calling rfft (DFT for
    real-valued signal)

    Parameters
    ----------
    F : np.ndarray of shape (2M,)
        vector of real and imaginary parts of a DFT result
    UF: np.ndarray of shape (2M,2M)
        covariance matrix associated with real and imaginary parts of F
    Nx: int, optional
        number of samples of iDFT result
    Cc: np.ndarray, optional
        cosine part of sensitivities (without scaling factor 1/N)
    Cs: np.ndarray, optional
        sine part of sensitivities (without scaling factor 1/N)
    returnC : bool, optional
        If True, return sensitivity matrix blocks (without scaling factor 1/N),
        if False do not return them

    Returns
    -------
    x : np.ndarray
        vector of time domain signal values
    Ux : np.ndarray
        covariance matrix associated with x
    Cc and Cs : Dict
        Keys are "Cc", "Cs" and values the respective sensitivity matrix entries

    References
    ----------
        * Eichstädt and Wilkens [Eichst2016]_

    Raises
    ------
    ValueError
        If Nx is not smaller than dimension of UF - 2
    """
    N = UF.shape[0] - 2

    if Nx is None:
        Nx = N
    elif Nx > UF.shape[0] - 2:
        raise ValueError(
            f"GUM_iDFT: Nx if provided is expected to be smaller or equal to number "
            f"of rows and columns of UF - 2, but UF is of shape {UF.shape} and Nx ="
            f" {Nx}."
        )

    beta = 2 * np.pi * np.arange(Nx) / N

    # calculate inverse DFT; Note: scaling factor 1/N is accounted for at the end
    x = np.fft.irfft(F[: N // 2 + 1] + 1j * F[N // 2 + 1 :])[:Nx]
    if not isinstance(Cc, np.ndarray):  # calculate sensitivities
        Cc = np.zeros((Nx, N // 2 + 1))
        Cc[:, 0] = 1.0
        Cc[:, -1] = np.cos(np.pi * np.arange(Nx))
        for k in range(1, N // 2):
            Cc[:, k] = 2 * np.cos(k * beta)

    if not isinstance(Cs, np.ndarray):
        Cs = np.zeros((Nx, N // 2 + 1))
        Cs[:, 0] = 0.0
        Cs[:, -1] = -np.sin(np.pi * np.arange(Nx))
        for k in range(1, N // 2):
            Cs[:, k] = -2 * np.sin(k * beta)

    # calculate blocks of uncertainty matrix
    if len(UF.shape) == 2:
        RR = UF[: N // 2 + 1, : N // 2 + 1]
        RI = UF[: N // 2 + 1, N // 2 + 1 :]
        II = UF[N // 2 + 1 :, N // 2 + 1 :]
        # propagate uncertainties
        Ux = np.dot(Cc, np.dot(RR, Cc.T))
        Ux = Ux + 2 * np.dot(Cc, np.dot(RI, Cs.T))
        Ux = Ux + np.dot(Cs, np.dot(II, Cs.T))
    else:
        RR = UF[: N // 2 + 1]
        II = UF[N // 2 + 1 :]
        Ux = np.dot(Cc, _prod(RR, Cc.T)) + np.dot(Cs, _prod(II, Cs.T))

    if returnC:
        return x, Ux / N ** 2, {"Cc": Cc, "Cs": Cs}
    else:
        return x, Ux / N ** 2


def GUM_DFTfreq(N, dt=1):
    """Return the Discrete Fourier Transform sample frequencies

    Parameters
    ----------
        N: int
            window length
        dt: float
            sample spacing (inverse of sampling rate)

    Returns
    -------
        f: ndarray
            Array of length ``n//2 + 1`` containing the sample frequencies

    See also
    --------
        `mod`::numpy.fft.rfftfreq

    """

    return np.fft.rfftfreq(N, dt)


def DFT2AmpPhase(
    F: np.ndarray,
    UF: np.ndarray,
    keep_sparse: Optional[bool] = False,
    tol: Optional[float] = 1.0,
    return_type: Optional[str] = "separate",
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Transformation from real and imaginary parts to magnitude and phase

    Calculate the matrix
    U_AP = [[U1,U2],[U2^T,U3]]
    associated with magnitude and phase of the vector F=[real,imag]
    with associated covariance matrix U_F=[[URR,URI],[URI^T,UII]]

    Parameters
    ----------
    F : np.ndarray of shape (2M,)
        vector of real and imaginary parts of a DFT result
    UF : np.ndarray of shape (2M,2M)
        covariance matrix associated with F
    keep_sparse : bool, optional
        if true then UAP will be sparse if UF is one-dimensional
    tol : float, optional
        lower bound for A/uF below which a warning will be issued
        concerning unreliable results
    return_type : str, optional
        If "separate" then magnitude and phase are returned as separate
        arrays A and P. Otherwise the list [A, P] is returned

    If ``return_type == separate``:

    Returns
    -------
    A : np.ndarray
        vector of magnitude values
    P : np.ndarray
        vector of phase values in radians, in the range [-pi, pi], but only
        present if ``return_type = 'separate'``
    UAP : np.ndarray
        covariance matrix associated with (A,P)

    Otherwise:

    Returns
    -------
    AP : np.ndarray
        vector of magnitude and phase values
    UAP : np.ndarray
        covariance matrix associated with AP
    """
    # calculate inverse DFT
    N = len(F) - 2
    R = F[: N // 2 + 1]
    I = F[N // 2 + 1 :]

    A = np.sqrt(R ** 2 + I ** 2)  # absolute value
    P = np.arctan2(I, R)  # phase value
    if len(UF.shape) == 1:
        uF = 0.5 * (
            np.sqrt(UF[: N // 2 + 1]) + np.sqrt(UF[N // 2 + 1 :])
        )  # uncertainty of real,imag
    else:
        uF = 0.5 * (
            np.sqrt(np.diag(UF[: N // 2 + 1, : N // 2 + 1]))
            + np.sqrt(np.diag(UF[N // 2 + 1 :, N // 2 + 1 :]))
        )
    if np.any(A / uF < tol):
        print(
            "DFT2AmpPhase Warning\n Some amplitude values are below the "
            "defined threshold.\nThe GUM formulas may become unreliable and a Monte "
            "Carlo approach is recommended instead.\nThe actual minimum value of A/uF"
            f"is {round((A / uF).min(), 2)} and the threshold is {round(tol, 2)}."
        )
    aR = R / A  # sensitivities
    aI = I / A
    pR = -I / A ** 2
    pI = R / A ** 2

    if len(UF.shape) == 1:  # uncertainty calculation of zero correlation
        URR = UF[: N // 2 + 1]
        UII = UF[N // 2 + 1 :]
        U11 = URR * aR ** 2 + UII * aI ** 2
        U12 = aR * URR * pR + aI * UII * pI
        U22 = URR * pR ** 2 + UII * pI ** 2
        UAP = sparse.diags([np.r_[U11, U22], U12, U12], [0, N // 2 + 1, -(N // 2 + 1)])
        if not keep_sparse:
            UAP = UAP.toarray()
    else:  # uncertainty calculation for full covariance
        URR = UF[: N // 2 + 1, : N // 2 + 1]
        URI = UF[: N // 2 + 1, N // 2 + 1 :]
        UII = UF[N // 2 + 1 :, N // 2 + 1 :]
        U11 = (
            _prod(aR, _prod(URR, aR))
            + _prod(aR, _prod(URI, aI))
            + _prod(aI, _prod(URI.T, aR))
            + _prod(aI, _prod(UII, aI))
        )
        U12 = (
            _prod(aR, _prod(URR, pR))
            + _prod(aR, _prod(URI, pI))
            + _prod(aI, _prod(URI.T, pR))
            + _prod(aI, _prod(UII, pI))
        )
        U22 = (
            _prod(pR, _prod(URR, pR))
            + _prod(pR, _prod(URI, pI))
            + _prod(pI, _prod(URI.T, pR))
            + _prod(pI, _prod(UII, pI))
        )
        UAP = np.vstack((np.hstack((U11, U12)), np.hstack((U12.T, U22))))

    if return_type == "separate":
        return A, P, UAP  # amplitude and phase as separate variables
    else:
        return np.r_[A, P], UAP


def AmpPhase2DFT(
    A: np.ndarray, P: np.ndarray, UAP: np.ndarray, keep_sparse: Optional[bool] = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Transformation from magnitude and phase to real and imaginary parts

    Calculate the vector F=[real,imag] and propagate the covariance matrix UAP
    associated with [A, P]

    Parameters
    ----------
    A : np.ndarray of shape (N,)
        vector of magnitude values
    P : np.ndarray of shape (N,)
        vector of phase values (in radians)
    UAP : np.ndarray of shape (2N,2N) or of shape (2N,)
        covariance matrix associated with (A,P) or vector of squared standard
        uncertainties [u^2(A),u^2(P)]
    keep_sparse : bool, optional
        whether to transform sparse matrix to numpy array or not

    Returns
    -------
    F : np.ndarray of shape (2N,)
        vector of real and imaginary parts of DFT result
    UF : np.ndarray of shape (2N,2N)
        covariance matrix associated with F

    Raises
    ------
    ValueError
        If dimensions of A, P and UAP do not match.
    """
    if len(A.shape) != 1:
        raise ValueError(
            f"AmpPhase2DFT: A is expected to be a 1-dimensional vector but is of shape"
            f" {A.shape}."
        )
    if A.shape != P.shape:
        raise ValueError(
            f"AmpPhase2DFT: The dimensions of A and P are expected to match but A is "
            f"of shape {A.shape} and P is of shape {P.shape}."
        )
    if UAP.shape != (2 * len(A), 2 * len(A)) and UAP.shape != (2 * len(A),):
        raise ValueError(
            f"AmpPhase2DFT: Each dimension of UAP is expected to be twice the length "
            f"of A, but A is of length {len(A)} and UAP is of shape {UAP.shape}."
        )

    # calculation of F
    F = np.r_[A * np.cos(P), A * np.sin(P)]  # calculation of best estimate

    # calculation of sensitivities
    CRA = np.cos(P)
    CRP = -A * np.sin(P)
    CIA = np.sin(P)
    CIP = A * np.cos(P)

    # assignment of uncertainty blocks in UAP
    N = len(A)
    if UAP.shape == (2 * N,):  # zero correlation; just standard deviations
        Ua = UAP[:N]
        Up = UAP[N:]
        U11 = CRA * Ua * CRA + CRP * Up * CRP
        U12 = CRA * Ua * CIA + CRP * Up * CIP
        U22 = CIA * Ua * CIA + CIP * Up * CIP
        UF = sparse.diags([np.r_[U11, U22], U12, U12], [0, N, -N])
        if not keep_sparse:
            UF = UF.toarray()
    else:
        if isinstance(UAP, sparse.dia_matrix):
            nrows = 2 * N
            offset = UAP.offsets
            diags = UAP.data
            Uaa = diags[0][:N]
            Uap = diags[1][offset[1] : nrows + offset[1]]
            Upp = diags[0][N:]

            U11 = Uaa * CRA ** 2 + CRP * Uap * CRA + CRA * Uap * CRP + Upp * CRP ** 2
            U12 = CRA * Uaa * CIA + CRP * Uap * CIA + CRA * Uap * CIA + CRP * Upp * CIP
            U22 = Uaa * CIA ** 2 + CIP * Uap * CIA + CIA * Uap * CIP + Upp * CIP ** 2

            UF = sparse.diags(
                [np.r_[U11, U22], U12, U12], [0, N, -N]
            )  # default is sparse
            if not keep_sparse:
                UF = UF.toarray()  # fall back to non-sparse
        else:
            Uaa = UAP[:N, :N]
            Uap = UAP[:N, N:]
            Upp = UAP[N:, N:]

            U11 = (
                _prod(CRA, _prod(Uaa, CRA))
                + _prod(CRP, _prod(Uap.T, CRA))
                + _prod(CRA, _prod(Uap, CRP))
                + _prod(CRP, _prod(Upp, CRP))
            )
            U12 = (
                _prod(CRA, _prod(Uaa, CIA))
                + _prod(CRP, _prod(Uap.T, CIA))
                + _prod(CRA, _prod(Uap, CIP))
                + _prod(CRP, _prod(Upp, CIP))
            )
            U22 = (
                _prod(CIA, _prod(Uaa, CIA))
                + _prod(CIP, _prod(Uap.T, CIA))
                + _prod(CIA, _prod(Uap, CIP))
                + _prod(CIP, _prod(Upp, CIP))
            )

            # stack together the full covariance matrix
            UF = np.vstack((np.hstack((U11, U12)), np.hstack((U12.T, U22))))

    return F, UF


def Time2AmpPhase(
    x: np.ndarray, Ux: Union[np.ndarray, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transformation from time domain to amplitude and phase via DFT

    Parameters
    ----------
    x : np.ndarray of shape (N,)
        time domain signal
    Ux : np.ndarray of shape (N,) or of shape (N,N) or float
        covariance matrix associated with x, or vector of squared standard
        uncertainties, or noise variance as float

    Returns
    -------
    A : np.ndarray
        amplitude values
    P : np.ndarray
        phase values
    UAP : np.ndarray
        covariance matrix associated with [A,P]
    """
    F, UF = GUM_DFT(x, Ux)  # propagate to DFT domain
    A, P, UAP = DFT2AmpPhase(F=F, UF=UF, return_type="separate")  # propagate to
    # amplitude and phase
    return A, P, UAP


def Time2AmpPhase_multi(x, Ux, selector=None):
    """Transformation from time domain to amplitude and phase

    Perform transformation for a set of M signals of the same type.

    Parameters
    ----------
        x: np.ndarray of shape (M, nx)
            M time domain signals of length nx
        Ux: np.ndarray of shape (M,)
            squared standard deviations representing noise variances of the
            signals x
        selector: np.ndarray of shape (L,), optional
            indices of amplitude and phase values that should be returned;
            default is 0:N-1
    Returns
    -------
        A: np.ndarray of shape (M,N)
            amplitude values
        P: np.ndarray of shape (M,N)
            phase values
        UAP: np.ndarray of shape (M, 3N)
            diagonals of the covariance matrices: [diag(UAA), diag(UAP),
            diag(UPP)]
    """
    M, nx = x.shape
    assert len(Ux) == M
    N = nx // 2 + 1
    if not isinstance(selector, np.ndarray):
        selector = np.arange(nx // 2 + 1)
    ns = len(selector)

    A = np.zeros((M, ns))
    P = np.zeros_like(A)
    UAP = np.zeros((M, 3 * ns))
    CxCos = None
    CxSin = None
    for m in range(M):
        F, UF, CX = GUM_DFT(x[m, :], Ux[m], CxCos, CxSin, returnC=True)
        CxCos = CX["CxCos"]
        CxSin = CX["CxSin"]
        A_m, P_m, UAP_m = DFT2AmpPhase(F, UF, keep_sparse=True)
        A[m, :] = A_m[selector]
        P[m, :] = P_m[selector]
        UAP[m, :ns] = UAP_m.data[0][:N][selector]
        UAP[m, ns : 2 * ns] = UAP_m.data[1][
            UAP_m.offsets[1] : 2 * N + UAP_m.offsets[1]
        ][selector]
        UAP[m, 2 * ns :] = UAP_m.data[0][N:][selector]

    return A, P, UAP


def AmpPhase2Time(
    A: np.ndarray, P: np.ndarray, UAP: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Transformation from amplitude and phase to time domain

    GUM propagation of covariance matrix UAP associated with DFT amplitude A and
    phase P to the result of the inverse DFT. Uncertainty UAP is assumed to be given
    for amplitude and phase with blocks:
    UAP = [[u(A,A), u(A,P)],[u(P,A),u(P,P)]]

    Parameters
    ----------
    A : np.ndarray of shape (N, )
        vector of amplitude values
    P : np.ndarray of shape (N, )
        vector of phase values (in rad)
    UAP : np.ndarray of shape (2N, 2N)
        covariance matrix associated with [A,P]

    Returns
    -------
    x : np.ndarray of shape (N, )
        vector of time domain values
    Ux : np.ndarray of shape (2N, 2N)
        covariance matrix associated with x

    Raises
    ------
    ValueError
        If dimension of UAP is not even.
    """
    N = UAP.shape[0] - 2
    if A.shape != P.shape:
        raise ValueError(
            f"AmpPhase2Time: The dimensions of A and P are expected to match but A is "
            f"of shape {A.shape} and P is of shape {P.shape}."
        )
    if np.mod(N, 2) != 0:
        raise ValueError(
            "AmpPhase2Time: Dimension of covariance matrix UAP is "
            f"supposed to be even, but UAP has {N+2} rows. Please "
            f"provide covariance matrix of dimension {len(A) + len(P)} x "
            f" {len(A) + len(P)}."
        )
    beta = 2 * np.pi * np.arange(N) / N

    # calculate inverse DFT
    F = A * np.exp(1j * P)
    x = np.fft.irfft(F)

    Pf = np.r_[P, -P[-2:0:-1]]  # phase values to take into account symmetric part
    Cc = np.zeros((N, N // 2 + 1))  # sensitivities wrt cosine part
    Cc[:, 0] = np.cos(P[0])
    Cc[:, -1] = np.cos(P[-1] + np.pi * np.arange(N))
    for k in range(1, N // 2):
        Cc[:, k] = 2 * np.cos(Pf[k] + k * beta)

    Cs = np.zeros((N, N // 2 + 1))  # sensitivities wrt sinus part
    Cs[:, 0] = -A[0] * np.sin(P[0])
    Cs[:, -1] = -A[-1] * np.sin(P[-1] + np.pi * np.arange(N))
    for k in range(1, N // 2):
        Cs[:, k] = -A[k] * 2 * np.sin(Pf[k] + k * beta)

    # calculate blocks of uncertainty matrix
    if len(UAP.shape) == 1:
        AA = UAP[: N // 2 + 1]
        PP = UAP[N // 2 + 1 :]
        Ux = np.dot(Cc, _prod(AA, Cc.T)) + np.dot(Cs, _prod(PP, Cs.T))
    else:
        if isinstance(UAP, sparse.dia_matrix):
            nrows = UAP.shape[0]
            n = nrows // 2
            offset = UAP.offsets
            diags = UAP.data
            AA = diags[0][:n]
            AP = diags[1][offset[1] : nrows + offset[1]]
            PP = diags[0][n:]
            Ux = (
                np.dot(Cc, _prod(AA, Cc.T))
                + 2 * np.dot(Cc, _prod(AP, Cs.T))
                + np.dot(Cs, _prod(PP, Cs.T))
            )
        else:
            AA = UAP[: N // 2 + 1, : N // 2 + 1]
            AP = UAP[: N // 2 + 1, N // 2 + 1 :]
            PP = UAP[N // 2 + 1 :, N // 2 + 1 :]
            # propagate uncertainties
            Ux = (
                np.dot(Cc, np.dot(AA, Cc.T))
                + 2 * np.dot(Cc, np.dot(AP, Cs.T))
                + np.dot(Cs, np.dot(PP, Cs.T))
            )

    return x, Ux / N ** 2


def DFT_transferfunction(X, Y, UX, UY):
    """Calculation of the transfer function H = Y/X in the frequency domain

    Calculate the transfer function with X being the Fourier transform
    of the system's input signal and Y that of the output signal.

    Parameters
    ----------
        X: np.ndarray
            real and imaginary parts of the system's input signal
        Y: np.ndarray
            real and imaginary parts of the system's output signal
        UX: np.ndarray
            covariance matrix associated with X
        UY: np.ndarray
            covariance matrix associated with Y

    Returns
    -------
        H: np.ndarray
            real and imaginary parts of the system's frequency response
        UH: np.ndarray
            covariance matrix associated with H

    This function only calls `DFT_deconv`.
    """
    return DFT_deconv(X, Y, UX, UY)


def DFT_deconv(
    H: np.ndarray, Y: np.ndarray, UH: np.ndarray, UY: np.ndarray
) -> Tuple[np.ndarray, Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
    """Deconvolution in the frequency domain

    GUM propagation of uncertainties for the deconvolution X = Y/H with Y and
    H being the Fourier transform of the measured signal and of the system's impulse
    response, respectively.

    This function returns the covariance matrix as a tuple of blocks if too large for
    complete storage in memory.

    Parameters
    ----------
    H : np.ndarray of shape (2M,)
        real and imaginary parts of frequency response values (M an even integer)
    Y : np.ndarray of shape (2M,)
        real and imaginary parts of DFT values
    UH : np.ndarray of shape (2M,2M) or (2M,)
        full covariance or diagonal of the covariance matrix associated with H
    UY : np.ndarray of shape (2M,2M) or (2M,)
        full covariance or diagonal of the covariance matrix associated with Y

    Returns
    -------
    X : np.ndarray of shape (2M,)
        real and imaginary parts of DFT values of deconv result
    UX : np.ndarray of shape (2M,2M) or 3-tuple of np.ndarray of shape (M,M)
        Covariance matrix associated with real and imaginary part of X. If the matrix
        fully assembled does not fit the memory, we return the auto-covariance for the
        real parts ``URRX``and the imaginary parts ``UIIX`` and the covariance between
        the real and imaginary parts ``URIX`` as separate
        :class:`np.ndarrays <numpy.ndarray>` arranged as follows: ``(URRX, URIX, UIIX)``

    References
    ----------
        * Eichstädt and Wilkens [Eichst2016]_

    Raises
    ------
    ValueError
        If dimensions of H, Y, UY and UH do not match accordingly.
    """
    if len(H) != len(Y):
        raise ValueError(
            f"The dimensions of H and Y are expected to match but H is of length "
            f"{len(H)} and Y is of length {len(Y)}."
        )

    N = len(UH) - 2
    if np.mod(N, 2) != 0:
        raise ValueError(
            "DFT_deconv: Dimension of covariance matrix UH is "
            f"supposed to be even, but UH has {N+2} rows. Please "
            f"provide covariance matrix of dimension {len(H)} x "
            f" {len(H)}."
        )

    if UY.shape not in ((len(Y), len(Y)), (len(Y),)):
        raise ValueError(
            f"DFT_deconv:Each dimension of UY is expected to match the length of "
            f"Y, but Y is of length {len(Y)} and UY is of shape {UY.shape}."
        )

    if UH.shape not in ((len(H), len(H)), (len(H),)):
        raise ValueError(
            f"DFT_deconv: The dimensions of UH and H are expected to match, "
            f"but H is of length {len(H)} and UH is of shape {UH.shape}."
        )

    # real and imaginary parts of system and signal
    rH, iH = H[: N // 2 + 1], H[N // 2 + 1 :]
    rY, iY = Y[: N // 2 + 1], Y[N // 2 + 1 :]

    Yc = Y[: N // 2 + 1] + 1j * Y[N // 2 + 1 :]
    Hc = H[: N // 2 + 1] + 1j * H[N // 2 + 1 :]
    X = np.r_[np.real(Yc / Hc), np.imag(Yc / Hc)]  # type: np.ndarray

    # sensitivities
    norm = rH ** 2 + iH ** 2
    RY = np.r_[rH / norm, iH / norm]
    IY = np.r_[-iH / norm, rH / norm]
    RH = np.r_[
        (-rY * rH ** 2 + rY * iH ** 2 - 2 * iY * iH * rH) / norm ** 2,
        (iY * rH ** 2 - iY * iH ** 2 - 2 * rY * rH * iH) / norm ** 2,
    ]
    IH = np.r_[
        (-iY * rH ** 2 + iY * iH ** 2 + 2 * rY * iH * rH) / norm ** 2,
        (-rY * rH ** 2 + rY * iH ** 2 - 2 * iY * rH * iH) / norm ** 2,
    ]
    # calculate blocks of uncertainty matrix
    URRX = _matprod(UY, RY, RY) + _matprod(UH, RH, RH)
    URIX = _matprod(UY, RY, IY) + _matprod(UH, RH, IH)
    UIIX = _matprod(UY, IY, IY) + _matprod(UH, IH, IH)

    try:
        UX = np.vstack((np.hstack((URRX, URIX)), np.hstack((URIX.T, UIIX))))  # type:
        # np.ndarray
    except MemoryError:
        print(
            "DFT_deconv: Could not put covariance matrix together due to memory "
            "constraints.\nReturning the three blocks (A,B,C) such that U = [[A,B],"
            "[B.T,C]] instead."
        )
        UX = (URRX, URIX, UIIX)
        # type: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

    return X, UX


def _matprod(
    M: np.ndarray, V: np.ndarray, W: np.ndarray, return_as_matrix: Optional[bool] = True
) -> np.ndarray:
    """Calculate the matrix-matrix-matrix product (V1,V2)M(W1,W2)

    Calculate the product for V=(V1,V2) and W=(W1,W2).

    Parameters
    ----------
    M : np.ndarray
        M can be sparse, one-dimensional or a full (quadratic) matrix.
    V, W : np.ndarray
        V=(V1,V2) and W=(W1,W2)

    Returns
    -------
    np.ndarray
        matrix product (V1,V2)M(W1,W2)

    Raises
    ------
    AssertionError
        If the dimensions of M, V and W do not match or M is a non-square matrix.
    """
    if len(M.shape) == 2:
        assert M.shape[0] == M.shape[1]
    assert M.shape[0] == V.shape[0]
    assert V.shape == W.shape
    N = V.shape[0] // 2
    v1 = V[:N]
    v2 = V[N:]
    w1 = W[:N]
    w2 = W[N:]
    if isinstance(M, sparse.dia_matrix):
        nrows = M.shape[0]
        offset = M.offsets
        diags = M.data
        A = diags[0][:N]
        B = diags[1][offset[1] : nrows + offset[1]]
        D = diags[0][N:]
        return np.diag(v1 * A * w1 + v2 * B * w1 + v1 * B * w2 + v2 * D * w2)
    elif len(M.shape) == 1:
        A = M[:N]
        D = M[N:]
        if return_as_matrix:
            return np.diag(v1 * A * w1 + v2 * D * w2)
        else:
            return np.r_[v1 * A * w1 + v2 * D * w2]
    else:
        A = M[:N, :N]
        B = M[:N, N:]
        D = M[N:, N:]
        return (
            _prod(v1, _prod(A, w1))
            + _prod(v2, _prod(B.T, w1))
            + _prod(v1, _prod(B, w2))
            + _prod(v2, _prod(D, w2))
        )


# for backward compatibility
GUMdeconv = DFT_deconv


def DFT_multiply(
    Y: np.ndarray, F: np.ndarray, UY: np.ndarray, UF: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Multiplication in the frequency domain

    GUM uncertainty propagation for multiplication in the frequency domain,
    where the second factor F may have an associated uncertainty. This method can be
    used, for instance, for the application of a low-pass filter in the frequency
    domain or the application of deconvolution as a multiplication with an inverse of
    known uncertainty.

    Parameters
    ----------
    Y : np.ndarray of shape (2M,)
        real and imaginary parts of the first factor
    F : np.ndarray of shape (2M,)
        real and imaginary parts of the second factor
    UY : np.ndarray either of shape (2M,) or of shape (2M,2M)
        covariance matrix or squared uncertainty associated with Y
    UF : np.ndarray of shape (2M,2M), optional
        covariance matrix associated with F

    Returns
    -------
    YF : np.ndarray of shape (2M,)
        the product of Y and F
    UYF : np.ndarray of shape (2M,2M)
        the uncertainty associated with YF

    Raises
    ------
    ValueError
        If dimensions of Y and F do not match.
    """
    if len(Y) != len(F):
        raise ValueError(
            f"GUM_multiply: The dimensions of Y and F are expected to match but Y is "
            f"of length {len(Y)} and F is of length {len(F)}."
        )

    def _calcU(A, UB):
        # uncertainty propagation for A*B with B uncertain (helper function)
        n = len(A)
        RA = A[: n // 2]
        IA = A[n // 2 :]
        if isinstance(UB, float):  # simpler calculation if only single uncertainty
            uRR = RA * UB * RA + IA * UB * IA
            uRI = RA * UB * IA - IA * UB * RA
            uII = IA * UB * IA + RA * UB * RA
        elif len(UB.shape) == 1:  # simpler calculation if no correlation
            UBRR = UB[: n // 2]
            UBII = UB[n // 2 :]
            uRR = RA * UBRR * RA + IA * UBII * IA
            uRI = RA * UBRR * IA - IA * UBII * RA
            uII = IA * UBRR * IA + RA * UBII * RA
        else:  # full calculation because of full input covariance
            UBRR = UB[: n // 2, : n // 2]
            UBRI = UB[: n // 2, n // 2 :]
            UBII = UB[n // 2 :, n // 2 :]
            uRR = (
                _prod(RA, _prod(UBRR, RA))
                - _prod(IA, _prod(UBRI.T, RA))
                - _prod(RA, _prod(UBRI, IA))
                + _prod(IA, _prod(UBII, IA))
            )
            uRI = (
                _prod(RA, _prod(UBRR, IA))
                - _prod(IA, _prod(UBRI.T, IA))
                + _prod(RA, _prod(UBRI, RA))
                - _prod(IA, _prod(UBII, RA))
            )
            uII = (
                _prod(IA, _prod(UBRR, IA))
                + _prod(RA, _prod(UBRI.T, IA))
                + _prod(IA, _prod(UBRI, RA))
                + _prod(RA, _prod(UBII, RA))
            )
        return uRR, uRI, uII

    N = len(Y)
    RY = Y[: N // 2]
    IY = Y[N // 2 :]  # decompose into block matrix
    RF = F[: N // 2]
    IF = F[N // 2 :]  # decompose into block matrix
    YF = np.r_[RY * RF - IY * IF, RY * IF + IY * RF]  # apply product rule
    if not isinstance(UF, np.ndarray):  # second factor is known exactly
        UYRR, UYRI, UYII = _calcU(F, UY)
        # Stack together covariance matrix
        UYF = np.vstack((np.hstack((UYRR, UYRI)), np.hstack((UYRI.T, UYII))))
    else:  # both factors are uncertain
        URR_Y, URI_Y, UII_Y = _calcU(F, UY)
        URR_F, URI_F, UII_F = _calcU(Y, UF)
        URR = URR_Y + URR_F
        URI = URI_Y + URI_F
        UII = UII_Y + UII_F
        # Stack together covariance matrix
        UYF = np.vstack((np.hstack((URR, URI)), np.hstack((URI.T, UII))))
    return YF, UYF
