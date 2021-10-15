"""
The :mod:`PyDynamic.misc.tools` module is a collection of miscellaneous helper
functions.

This module contains the following functions:

* :func:`FreqResp2RealImag`: Calculate real and imaginary parts from frequency
  response
* :func:`is_2d_matrix`: Check if a np.ndarray is a matrix
* :func:`is_2d_square_matrix`: Check if a np.ndarray is a two-dimensional square matrix
* :func:`is_vector`: Check if a np.ndarray is a vector
* :func:`make_semiposdef`: Make quadratic matrix positive semi-definite
* :func:`normalize_vector_or_matrix`: Scale an array of numbers to the interval between
  zero and one
* :func:`number_of_rows_equals_vector_dim`: Check if a matrix and a vector match in size
* :func:`plot_vectors_and_covariances_comparison`: Plot two vectors and their
  covariances side-by-side for visual comparison
* :func:`print_mat`: Print matrix (2D array) to the console or return as formatted
  string
* :func:`print_vec`: Print vector (1D array) to the console or return as formatted
  string
* :func:`progress_bar`: A simple and reusable progress-bar
* :func:`shift_uncertainty`: Shift the elements in the vector x and associated
  uncertainties ux
* :func:`trimOrPad`: trim or pad (with zeros) a vector to desired length
"""
from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
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
    "plot_vectors_and_covariances_comparison",
    "is_2d_square_matrix",
    "normalize_vector_or_matrix",
]


def shift_uncertainty(x: np.ndarray, ux: np.ndarray, shift: int):
    """Shift the elements in the vector x and associated uncertainties ux

    This function uses :func:`numpy.roll` to shift the elements in x
    and ux. See the linked official documentation for details.

    Parameters
    ----------
    x : np.ndarray of shape (N,)
        vector of estimates
    ux : float, np.ndarray of shape (N,) or of shape (N,N)
        uncertainty associated with the vector of estimates
    shift : int
        amount of shift

    Returns
    -------
    shifted_x : (N,) np.ndarray
        shifted vector of estimates
    shifted_ux : float, np.ndarray of shape (N,) or of shape (N,N)
        uncertainty associated with the shifted vector of estimates

    Raises
    ------
    ValueError
        If shift, x or ux are of unexpected type, dimensions of x and ux do not fit
        or ux is of unexpected shape
    """
    shifted_x = _shift_vector(vector=x, shift=shift)

    if isinstance(ux, float):
        return shifted_x, ux

    if isinstance(ux, np.ndarray):
        if is_vector(ux):
            return shifted_x, _shift_vector(vector=ux, shift=shift)
        elif is_2d_square_matrix(ux):
            shifted_ux = _shift_2d_matrix(ux, shift)
            return shifted_x, shifted_ux
        raise ValueError(
            "shift_uncertainty: input uncertainty ux is expected to be a vector or "
            f"two-dimensional, square matrix but is of shape {ux.shape}."
        )

    raise ValueError(
        "shift_uncertainty: input uncertainty ux is expected to be a float or a "
        f"numpy.ndarray but is of type {type(ux)}."
    )


def _cast_shift_to_int(shift: Any) -> int:
    try:
        return int(shift)
    except ValueError:
        raise ValueError(
            "shift_uncertainty: shift is expected to be type int or at least "
            f"cast-able to int, but is {shift} of type {type(shift)}. Please provide "
            "a valid value."
        )


def _shift_vector(vector: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(vector, shift)


def _shift_2d_matrix(matrix: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(matrix, (shift, shift), axis=(0, 1))


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
    """Print vector (1D array) to the console or return as formatted string

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
    """Print matrix (2D array) to the console or return as formatted string

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
        e = np.min(np.real(eigs(matrix, which="SR", return_eigenvectors=False)))
        count = 0
        # increase the eigenvalues until matrix is positive semi-definite
        while e < tol and count < maxiter:
            matrix += (np.absolute(e) + tol) * eye(n, format=matrix.format)
            e = np.min(np.real(eigs(matrix, which="SR", return_eigenvectors=False)))
            count += 1
        e = np.min(np.real(eigs(matrix, which="SR", return_eigenvectors=False)))
    # same procedure for non-sparse matrices
    else:
        matrix = 0.5 * (matrix + matrix.T)
        count = 0
        e = np.min(np.real(np.linalg.eigvals(matrix)))
        while e < tol and count < maxiter:
            e = np.min(np.real(np.linalg.eigvals(matrix)))
            matrix += (np.absolute(e) + tol) * np.eye(n)
        e = np.min(np.real(np.linalg.eigvals(matrix)))
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
    """Check if a np.ndarray is a matrix, i.e. is of shape (n,m)

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


def plot_vectors_and_covariances_comparison(
    vector_1: np.ndarray,
    vector_2: np.ndarray,
    covariance_1: np.ndarray,
    covariance_2: np.ndarray,
    title: Optional[str] = "Comparison between two vectors and corresponding "
    "uncertainties",
    label_1: Optional[str] = "vector_1",
    label_2: Optional[str] = "vector_2",
):
    """Plot two vectors and their covariances side-by-side for visual comparison

    Parameters
    ----------
    vector_1 : np.ndarray
        the first vector to compare
    vector_2 : np.ndarray
        the second vector to compare
    covariance_1 : np.ndarray
        the first covariance matrix to compare
    covariance_2 : np.ndarray
        the second covariance matrix to compare
    title : str, optional
        the title for the comparison plot, defaults to `"Comparison between two vectors
        and corresponding uncertainties"`
    label_1 : str, optional
        the label for the first vector in the legend and title for the first
        covariance plot, defaults to "vector_1"
    label_2 : str, optional
        the label for the second vector in the legend and title for the second
        covariance plot, defaults to "vector_2"
    """
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(title)
    ax[0][0].imshow(covariance_1)
    ax[0][0].set_title(label_1 + " uncertainties")
    ax[0][1].imshow(covariance_2)
    ax[0][1].set_title(label_2 + " uncertainties")
    ax[1][0].plot(vector_1, label=label_1)
    ax[1][0].plot(vector_2, label=label_2)
    ax[1][0].legend()
    ax[1][0].set_title(label_1 + " and " + label_2)
    ax[1][1].imshow(covariance_2 - covariance_1, norm=Normalize())
    ax[1][1].set_title("Relative difference of uncertainties")
    plt.show()


def is_2d_square_matrix(ndarray: np.ndarray) -> bool:
    """Check if a np.ndarray is a two-dimensional square matrix, i.e. is of shape (n,n)

    Parameters
    ----------
    ndarray : np.ndarray
        the array to check

    Returns
    -------
    bool
        True, if the array expands over exactly two dimensions of similar size,
        False otherwise
    """
    return is_2d_matrix(ndarray) and ndarray.shape[0] == ndarray.shape[1]


def normalize_vector_or_matrix(numbers: np.ndarray) -> np.ndarray:
    """Scale an array of numbers to the interval between zero and one

    If all values in the array are the same, the output array will be constant zero.

    Parameters
    ----------
    numbers : np.ndarray
        the :class:`numpy.ndarray` to normalize

    Returns
    -------
    np.ndarray
        the normalized array
    """
    minimum = translator = np.min(numbers)
    array_span = np.max(numbers) - minimum
    normalizer = array_span or 1.0
    return (numbers - translator) / normalizer
