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
import scipy.sparse as sparse


def col_hstack(vectors):
    """
    From tuple of 1D ndarrays make a 2D ndarray where the tuple
    elements are as column vectors horizontally stacked
    
    :param vectors: list of K 1D-ndarrays of dimension N
    :returns matrix: 2D ndarray of shape (N,K)
    
    """

    if isinstance(vectors,list):
        col_vectors = map(lambda x: x[:,np.newaxis], vectors)
    else:
        raise ValueError("Input must be of type list\n")    
    
    return np.hstack(col_vectors)


def find(assertions):
    """
    MATLAB-like determination of occurrence of assertion in an array using the
    numpy nonzero function
    """
    if not isinstance(assertions, tuple):
        raise ValueError("Input to 'find' needs to be a tuple.")

    inds = np.arange(len(assertions[0]))
    for assertion in assertions:
        inds = inds[np.nonzero(assertion[inds])[0]]

    return inds

def zerom(shape):
    """ Generate a numpy.matrix of zeros of given shape"""
    from numpy import zeros, matrix
    return matrix(zeros(shape))


def stack(elements):
    def make_matrix(v):
        if len(v.shape()) > 1:
            return v
        else:
            return v[:, np.newaxis]

    return np.hstack(map(lambda x: make_matrix(x), elements))


def print_vec(vector, prec=5, retS=False, vertical=False):
    """
    Print vector (!D array) to the command line of return as formatted string

    Parameters
    ----------
        vector : 1D nparray of shape (M,)
        prec : integer specifying the precision of the output
        vertical : boolean if print out vertical or not 
        retS : boolean if print or return string

    Returns
    -------
        string if retS is True

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
    Print matrix (2D array) to the command line or return as formatted string
    
    Parameters
    ----------
        matrix : 2D nparray of shape (M,N)
        prec : integer specifying the precision of the output
        vertical : boolean if print out vertical or not 
        retS : boolean if print or return string
         
    Returns
    -------
        string if retS is True

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
        maxiter: integer, the maximum number of iterations for increasing the
                 eigenvalues
        tol: float, tolerance for deciding if pos. semi-def.
        
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
    """
    Calculation of real and imaginary parts from amplitude and phase with
    associated uncertainties

    Parameters
    ----------

        Abs: ndarray of shape N - amplitude values
        Phase: ndarray of shape N - phase values in rad
        Unc: ndarray of shape 2Nx2N or 2N - uncertainties

    Returns
    -------

        Re,Im: ndarrays of shape N - real and imaginary parts (best estimate)
        URI: ndarray of shape 2Nx2N - uncertainties assoc. with Re and Im
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


def make_equidistant(t, y, uy, dt=50, kind='previous'):
    """
    Convert non-equidistant time series to equidistant by interpolation (WIP)

    Parameters
    ----------
        t: (N,) array_like
            timestamps
        y: (N,) array_like
            measurement values
        uy: float
            measurement values' uncertainty
        dt: float, optional
            desired interval length
        kind: str or int, optional
            Specifies the kind of interpolation as a string
            ('linear' or 'previous'; 'previous' simply returns the previous
            value of the point).
            Default is 'previous'.

    Returns
    -------
        t: (N,) array_like
            timestamps
        y: (N,) array_like
            measurement values
        uy: float
            measurement values' uncertainty
    """

    return t, y, uy
