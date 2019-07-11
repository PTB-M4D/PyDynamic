# -*- coding: utf-8 -*-
"""
Collection of miscellaneous helper functions.

This module contains the following functions:

* print_mat: print matrix (2D array) to the command line
* print_vec: print vector (1D array) to the command line
* make_semiposdef: Make quadratic matrix positive semi-definite by increasing
its eigenvalues
* FreqResp2RealImag: Calculation of real and imaginary parts from amplitude
and phase with associated uncertainties
* make_equidistant: Convert non-equidistant time series to equidistant by
interpolation
"""

__all__ = ['print_mat', 'print_vec', 'make_semiposdef', 'FreqResp2RealImag',
           'make_equidistant']

import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d


def print_vec(vector, prec=5, retS=False, vertical=False):
    """
    Print vector (!D array) to the console of return as formatted string

    Parameters
    ----------
        vector : 1D nparray of shape (M,)
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
    """
    Print matrix (2D array) to the console or return as formatted string
    
    Parameters
    ----------
        matrix : 2D nparray of shape (M,N)
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
        matrix : 2D nparray of shape (N,N)
        maxiter: int
            the maximum number of iterations for increasing the eigenvalues
        tol: float
            tolerance for deciding if pos. semi-def.
        verbose: bool
            If True print some more detail about input parameters.
        
    Returns
    -------
        nparray of shape (N,N)

    """
    n, m = matrix.shape
    if n != m:
        raise ValueError("Matrix has to be quadratic")
    # use specialised functions for sparse matrices
    if sparse.issparse(matrix):
        # enforce symmetric matrix
        matrix = 0.5 * (matrix + matrix.T)
        # calculate smallest eigenvalue
        e = np.real(sparse.eigs(matrix, which="SR",
                                return_eigenvectors=False)).min()
        count = 0
        # increase the eigenvalues until matrix is positive semi-definite
        while e < tol and count < maxiter:
            matrix += (np.absolute(e) + tol) * sparse.eye(n,
                                                          format=matrix.format)
            e = np.real(sparse.eigs(matrix, which="SR",
                                    return_eigenvectors=False)).min()
            count += 1
        e = np.real(
            sparse.eigs(matrix, which="SR", return_eigenvectors=False)).min()
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

    Calculation of real and imaginary parts from amplitude and phase with
    associated uncertainties

    Parameters
    ----------

        Abs: ndarray of shape N
            amplitude values
        Phase: ndarray of shape N
            phase values in rad
        Unc: ndarray of shape 2Nx2N or 2N
            uncertainties
        MCruns: bool
            Iterations for Monte Carlo simulation

    Returns
    -------

        Re,Im: ndarrays of shape N
            real and imaginary parts (best estimate)
        URI: ndarray of shape 2Nx2N
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


def make_equidistant(t, y, uy, dt=5e-2, kind='previous'):
    """
    Convert non-equidistant time series to equidistant by interpolation (WIP)

    Parameters
    ----------
        t: (N,) array_like
            timestamps
        y: (N,) array_like
            measurement values
        uy: (N,) array_like
            measurement values' uncertainties
        dt: float, optional
            desired interval length in seconds
        kind: str or int, optional
            Specifies the kind of interpolation as a string
            ('linear' or 'previous'; 'previous' simply returns the previous
            value of the point). Default is 'previous'.

    Returns
    -------
        t_new: (N,) array_like
            timestamps
        y_new: (N,) array_like
            measurement values
        uy_new: (N,) array_like
            measurement values' uncertainties

    References
    ----------
        * White [White2017]_
    """
    # Setup new vectors of timestamps, measurement values and uncertainties.
    t_new = np.arange(t[0], t[-1], dt)
    y_new = np.zeros_like(t_new)
    uy_new = np.zeros_like(t_new)

    if kind == 'previous':
        # Compute each previous measurement value and uncertainty by
        # iterating over t_new as ndarray.
        it = np.nditer(t_new, flags=['f_index'])
        while not it.finished:
            # Find measurement value and uncertainty for biggest of all
            # timestamps smaller or equal than current time.
            last_index = np.where(t <= it[0])[0][-1]
            y_new[it.index] = y[last_index]
            uy_new[it.index] = uy[last_index]
            it.iternext()
    else:
        if kind == 'linear':
            # Linearly interpolate each new measurement value and uncertainty by
            # iterating over t_new as ndarray.
            t_interpolant = interp1d(t, y)
            y_new = t_interpolant(t_new)
        else:
            raise NotImplementedError

    return t_new, y_new, uy_new
