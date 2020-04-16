# -*- coding: utf-8 -*-
"""
Collection of miscellaneous helper functions.

This module contains the following functions:

* *print_vec*: Print vector (1D array) to the console or return as formatted
  string
* *print_mat*: Print matrix (2D array) to the console or return as formatted
  string
* *make_semiposdef*: Make quadratic matrix positive semi-definite
* *FreqResp2RealImag*: Calculate real and imaginary parts from frequency
  response
* *make_equidistant*: Interpolate non-equidistant time series to equidistant
* *trimOrPad*: trim or pad (with zeros) a vector to desired length
* *progress_bar*: A simple and reusable progress-bar
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import issparse, eye
from scipy.sparse.linalg.eigen.arpack import eigs
import sys

__all__ = ['print_mat', 'print_vec', 'make_semiposdef', 'FreqResp2RealImag',
           'make_equidistant', 'trimOrPad', 'progress_bar']


def trimOrPad(array, length, mode="constant"):

    if len(array) < length: # pad zeros to the right if too short
        return np.pad(array, (0,length - len(array)), mode=mode)
    else:                   # trim to given length otherwise
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
        [print_vec(matrix[k, :], prec=prec, vertical=False, retS=True) + "\n"
         for k in range(matrix.shape[0])])

    if retS:
        return s
    else:
        print(s)


def make_semiposdef(matrix, maxiter=10, tol=1e-12, verbose=False):
    """
    Make quadratic matrix positive semi-definite by increasing its eigenvalues

    Parameters
    ----------
        matrix : (N,N) array_like
        maxiter: int
            the maximum number of iterations for increasing the eigenvalues
        tol: float
            tolerance for deciding if pos. semi-def.
        verbose: bool
            If True print some more detail about input parameters.

    Returns
    -------
        (N,N) array_like
            quadratic positive semi-definite matrix

    """
    n, m = matrix.shape
    if n != m:
        raise ValueError("Matrix has to be quadratic")
    # use specialised functions for sparse matrices
    if issparse(matrix):
        # enforce symmetric matrix
        matrix = 0.5 * (matrix + matrix.T)
        # calculate smallest eigenvalue
        e = np.real(eigs(matrix, which="SR",
                         return_eigenvectors=False)).min()
        count = 0
        # increase the eigenvalues until matrix is positive semi-definite
        while e < tol and count < maxiter:
            matrix += (np.absolute(e) + tol) * eye(n, format=matrix.format)
            e = np.real(eigs(matrix, which="SR",
                             return_eigenvectors=False)).min()
            count += 1
        e = np.real(eigs(matrix, which="SR",
                         return_eigenvectors=False)).min()
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


def FreqResp2RealImag(Abs, Phase, Unc, MCruns=1e4):
    """ Calculate real and imaginary parts from frequency response

    Calculate real and imaginary parts from amplitude and phase with
    associated uncertainties.

    Parameters
    ----------

        Abs: (N,) array_like
            amplitude values
        Phase: (N,) array_like
            phase values in rad
        Unc: (2N, 2N) or (2N,) array_like
            uncertainties
        MCruns: bool
            Iterations for Monte Carlo simulation

    Returns
    -------

        Re, Im: (N,) array_like
            real and imaginary parts (best estimate)
        URI: (2N, 2N) array_like
            uncertainties assoc. with Re and Im
    """

    if len(Abs) != len(Phase) or 2 * len(Abs) != len(Unc):
        raise ValueError('\nLength of inputs are inconsistent.')

    if len(Unc.shape) == 1:
        Unc = np.diag(Unc)

    Nf = len(Abs)

    AbsPhas = np.random.multivariate_normal(np.hstack((Abs, Phase)), Unc,
                                            int(MCruns))  # draw MC inputs

    H = AbsPhas[:, :Nf] * np.exp(
        1j * AbsPhas[:, Nf:])  # calculate complex frequency response values
    RI = np.hstack((np.real(H), np.imag(H)))  # transform to real, imag

    Re = np.mean(RI[:, :Nf])
    Im = np.mean(RI[:, Nf:])
    URI = np.cov(RI, rowvar=False)

    return Re, Im, URI


def make_equidistant(t, y, uy, dt=5e-2, kind="linear"):
    """ Interpolate non-equidistant time series to equidistant

    Interpolate measurement values and propagate uncertainties accordingly.

    Parameters
    ----------
        t: (N,) array_like
            timestamps in ascending order
        y: (N,) array_like
            corresponding measurement values
        uy: (N,) array_like
            corresponding measurement values' uncertainties
        dt: float, optional
            desired interval length in seconds
        kind: str, optional
            Specifies the kind of interpolation for the measurement values
            as a string ('previous', 'next', 'nearest' or 'linear').

    Returns
    -------
        t_new : (N,) array_like
            interpolation timestamps
        y_new : (N,) array_like
            interpolated measurement values
        uy_new : (N,) array_like
            interpolated measurement values' uncertainties

    References
    ----------
        * White [White2017]_
    """
    from ..uncertainty.interpolation import interp1d_unc

    # Setup new vector of timestamps.
    t_new = np.arange(t[0], t[-1], dt)

    return interp1d_unc(t_new, t, y, uy, kind)



def progress_bar(count, count_max, width=30, prefix="", done_indicator="#", todo_indicator =".", fout=sys.stdout):
    """
    A simple and reusable progress-bar

    Parameters
    ----------
        count: int
            current status of iterations, assumed to be zero-based
        count_max: int
            total number of iterations
        width: int, optional
            width of the actual progressbar (actual printed line will be wider)
        prefix: str, optional
            some text that will be printed in front of the bar (i.e.
            "Progress of ABC:")
        done_indicator: str, optional
            what character is used as "already-done"-indicator
        todo_indicator: str, optional
            what character is used as "not-done-yet"-indicator
        fout: file-object, optional
            where the progress-bar should be written/printed to
    """
    x = int(width * (count+1) / count_max)
    progressString = "{PREFIX}[{DONE}{NOTDONE}] {COUNT}/{COUNTMAX}\r".format(
        PREFIX=prefix,
        DONE=x * done_indicator,
        NOTDONE=(width-x) * todo_indicator,
        COUNT=count+1,
        COUNTMAX=count_max)

    fout.write(progressString)
