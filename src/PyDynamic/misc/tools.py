# -*- coding: utf-8 -*-
"""
The :mod:`PyDynamic.misc.tools` module is a collection of miscellaneous helper
functions.

This module contains the following functions:

* :func:`print_vec`: Print vector (1D array) to the console or return as formatted
  string
* :func:`print_mat`: Print matrix (2D array) to the console or return as formatted
  string
* :func:`make_semiposdef`: Make quadratic matrix positive semi-definite
* :func:`FreqResp2RealImag`: Calculate real and imaginary parts from frequency
  response
* :func:`make_equidistant`: Interpolate non-equidistant time series to equidistant
* :func:`trimOrPad`: trim or pad (with zeros) a vector to desired length
* :func:`progress_bar`: A simple and reusable progress-bar
* :func:`is_vector`: Check if a np.ndarray is a vector
* :func:`is_2d_matrix`: Check if a np.ndarray is a matrix
* :func:`number_of_rows_equals_vector_dim`: Check if a matrix and a vector match in size
"""
from typing import Optional

import numpy as np
from scipy.sparse import eye, issparse
from scipy.sparse.linalg.eigen.arpack import eigs

__all__ = [
    "print_mat",
    "print_vec",
    "make_semiposdef",
    "FreqResp2RealImag",
    "make_equidistant",
    "trimOrPad",
    "progress_bar",
    "shift_uncertainty",
    "is_vector",
    "is_2d_matrix",
    "number_of_rows_equals_vector_dim",
]

def shift_uncertainty(x, ux, shift):
    """Shift the elements in the vector x (and associated uncertainty ux) by shift elements.
        This method uses :class:`numpy.roll` to shift the elements in x and ux. See documentation of np.roll for details.

    Parameters
    ----------
        x: (N,) array_like
            vector of estimates
        ux: float, np.ndarray of shape (N,) or of shape (N,N)
            uncertainty associated with the vector of estimates
        shift: int
            amount of shift

    Returns
    -------
        xs: (N,) array
            shifted vector of estimates
        uxs: float, np.ndarray of shape (N,) or of shape (N,N)
            uncertainty associated with the shifted vector of estimates
    """

    assert(isinstance(shift, int))
    # application of shift to the vector of estimates
    xs = np.roll(x, shift)

    if isinstance(ux, float):       # no shift necessary for ux
        return xs, ux
    if isinstance(ux, np.ndarray):
        if len(ux.shape) == 1:      # uncertainties given as vector
            return xs, np.roll(ux, shift)
        elif len(ux.shape) == 2:      # full covariance matrix
            assert(ux.shape[0]==ux.shape[1])
            uxs = np.roll(ux, (shift, shift), axis=(0,1))
            return xs, uxs
        else:
            raise TypeError("Input uncertainty has incompatible shape")
    else:
        raise TypeError("Input uncertainty has incompatible type")

def trimOrPad(array, length, mode="constant"):
    """Trim or pad (with zeros) a vector to the desired length

    Parameters
    ----------
    array : list, 1D np.ndarray
        original data
    length : int
        length of output
    mode : str, optional
        handed over to np.pad, default "constant"

    Returns
    -------
    array_modified : np.ndarray of shape (length,)
        An array that is either trimmed or zero-padded to achieve
        the required `length`. Both actions are applied to the
        right side of the array
    """

    if len(array) < length:  # pad zeros to the right if too short
        return np.pad(array, (0, length - len(array)), mode=mode)
    else:  # trim to given length otherwise
        return array[0:length]


def print_vec(vector, prec=5, retS=False, vertical=False):
    """ Print vector (1D array) to the console or return as formatted string

    Parameters
    ----------
        vector : (M,) array_like
        prec : int
            the precision of the output
        vertical : bool
            print out vertical or not
        retS : bool
            print or return string

    Returns
    -------
        s : str
            if retS is True

    """
    if vertical:
        t = "\n"
    else:
        t = "\t"
    s = "".join(["%1.*g %s" % (int(prec), s, t) for s in vector])
    if retS:
        return s
    else:
        print(s)


def print_mat(matrix, prec=5, vertical=False, retS=False):
    """ Print matrix (2D array) to the console or return as formatted string

    Parameters
    ----------
        matrix : (M,N) array_like
        prec : int
            the precision of the output
        vertical : bool
            print out vertical or not
        retS : bool
            print or return string

    Returns
    -------
        s : str
            if retS is True

    """
    if vertical:
        matrix = matrix.T

    s = "".join(
        [
            print_vec(matrix[k, :], prec=prec, vertical=False, retS=True) + "\n"
            for k in range(matrix.shape[0])
        ]
    )

    if retS:
        return s
    else:
        print(s)


def make_semiposdef(
    matrix: np.ndarray,
    maxiter: Optional[int] = 10,
    tol: Optional[float] = 1e-12,
    verbose: Optional[bool] = False,
) -> np.ndarray:
    """Make quadratic matrix positive semi-definite by increasing its eigenvalues

    Parameters
    ----------
    matrix : array_like of shape (N,N)
        the matrix to process
    maxiter : int, optional
        the maximum number of iterations for increasing the eigenvalues, defaults to 10
    tol : float, optional
        tolerance for deciding if pos. semi-def., defaults to 1e-12
    verbose : bool, optional
        If True print smallest eigenvalue of the resulting matrix, if False (default)
        be quiet

    Returns
    -------
    (N,N) array_like
        quadratic positive semi-definite matrix

    Raises
    ------
    ValueError
        If matrix is not square.
    """
    n, m = matrix.shape
    if n != m:
        raise ValueError("Matrix has to be quadratic")
    # use specialised functions for sparse matrices
    if issparse(matrix):
        # enforce symmetric matrix
        matrix = 0.5 * (matrix + matrix.T)
        # calculate smallest eigenvalue
        e = np.real(eigs(matrix, which="SR", return_eigenvectors=False)).min()
        count = 0
        # increase the eigenvalues until matrix is positive semi-definite
        while e < tol and count < maxiter:
            matrix += (np.absolute(e) + tol) * eye(n, format=matrix.format)
            e = np.real(eigs(matrix, which="SR", return_eigenvectors=False)).min()
            count += 1
        e = np.real(eigs(matrix, which="SR", return_eigenvectors=False)).min()
    # same procedure for non-sparse matrices
    else:
        matrix = 0.5 * (matrix + matrix.T)
        count = 0
        e = np.real(np.linalg.eigvals(matrix)).min()
        while e < tol and count < maxiter:
            e = np.real(np.linalg.eigvals(matrix)).min()
            matrix += (np.absolute(e) + tol) * np.eye(n)
        e = np.real(np.linalg.eigvals(matrix)).min()
    if verbose:
        print("Final result of make_semiposdef: smallest eigenvalue is %e" % e)
    return matrix


def FreqResp2RealImag(
    Abs: np.ndarray, Phase: np.ndarray, Unc: np.ndarray, MCruns: Optional[int] = 1000
):
    """Calculate real and imaginary parts from frequency response

    Calculate real and imaginary parts from amplitude and phase with
    associated uncertainties.

    Parameters
    ----------
    Abs : (N,) array_like
        amplitude values
    Phase : (N,) array_like
        phase values in rad
    Unc : (2N, 2N) or (2N,) array_like
        uncertainties either as full covariance matrix or as its main diagonal
    MCruns : int, optional
        number of iterations for Monte Carlo simulation, defaults to 1000

    Returns
    -------
    Re, Im : (N,) array_like
        best estimate of real and imaginary parts
    URI : (2N, 2N) array_like
        uncertainties assoc. with Re and Im
    """

    if len(Abs) != len(Phase) or 2 * len(Abs) != len(Unc):
        raise ValueError("\nLength of inputs are inconsistent.")

    if len(Unc.shape) == 1:
        Unc = np.diag(Unc)

    Nf = len(Abs)

    AbsPhas = np.random.multivariate_normal(
        np.hstack((Abs, Phase)), Unc, int(MCruns)
    )  # draw MC inputs

    H = AbsPhas[:, :Nf] * np.exp(
        1j * AbsPhas[:, Nf:]
    )  # calculate complex frequency response values
    RI = np.hstack((np.real(H), np.imag(H)))  # transform to real, imag

    Re = np.mean(RI[:, :Nf])
    Im = np.mean(RI[:, Nf:])
    URI = np.cov(RI, rowvar=False)

    return Re, Im, URI


def make_equidistant(*args, **kwargs):
    import warnings

    from ..uncertainty.interpolate import make_equidistant

    warnings.warn(
        "The method :mod:`PyDynamic.misc.tools.make_equidistant` will be moved "
        "to :mod:`PyDynamic.uncertainty.interpolate.make_equidistant` in the next "
        "major release 2.0.0. From version 1.4.3 on you should only use the new method "
        "instead. Please change 'from PyDynamic.misc.tools import make_equidistant' to "
        "'from PyDynamic.uncertainty.interpolate import make_equidistant'.",
        PendingDeprecationWarning,
    )
    return make_equidistant(*args, **kwargs)


def progress_bar(
    count,
    count_max,
    width: Optional[int] = 30,
    prefix: Optional[str] = "",
    done_indicator: Optional[str] = "#",
    todo_indicator: Optional[str] = ".",
    fout: Optional = None,
):
    """A simple and reusable progress-bar

    Parameters
    ----------
    count : int
        current status of iterations, assumed to be zero-based
    count_max : int
        total number of iterations
    width : int, optional
        width of the actual progressbar (actual printed line will be wider), default to
        30
    prefix : str, optional
        some text that will be printed in front of the bar (i.e.
        "Progress of ABC:"), if not given only progressbar itself will be printed
    done_indicator : str, optional
        what character is used as "already-done"-indicator, defaults to "#"
    todo_indicator : str, optional
        what character is used as "not-done-yet"-indicator, defaults to "."
    fout : file-object, optional
        where the progress-bar should be written/printed to, defaults to direct print
        to stdout
    """
    x = int(width * (count + 1) / count_max)
    progressString = "{PREFIX}[{DONE}{NOTDONE}] {COUNT}/{COUNTMAX}\r".format(
        PREFIX=prefix,
        DONE=x * done_indicator,
        NOTDONE=(width - x) * todo_indicator,
        COUNT=count + 1,
        COUNTMAX=count_max,
    )
    if fout is not None:
        fout.write(progressString)
    else:
        print(progressString)


def is_vector(ndarray: np.ndarray) -> bool:
    """Check if a np.ndarray is a vector, i.e. is of shape (n,)

    Parameters
    ----------
    ndarray : np.ndarray
        the array to check

    Returns
    -------
    bool
        True, if the array expands over one dimension only, False otherwise
    """
    return len(ndarray.shape) == 1


def is_2d_matrix(ndarray: np.ndarray) -> bool:
    """Check if a np.ndarray is a matrix, i.e. it expands over exactly two dimensions

    Parameters
    ----------
    ndarray : np.ndarray
        the array to check

    Returns
    -------
    bool
        True, if the array expands over exactly two dimensions, False otherwise
    """
    return len(ndarray.shape) == 2


def number_of_rows_equals_vector_dim(matrix: np.ndarray, vector: np.ndarray) -> bool:
    """Check if a matrix has the same number of rows as a vector

    Parameters
    ----------
    matrix : np.ndarray
        the matrix, that is supposed to have the same number of rows
    vector : np.ndarray
        the vector, that is supposed to have the same number of elements

    Returns
    -------
    bool
        True, if the number of rows coincide, False otherwise
    """
    return len(vector) == matrix.shape[0]
