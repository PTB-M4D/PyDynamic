# -*- coding: utf-8 -*-
"""
Collection of miscelleneous helper functions.

"""

import numpy as np

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
    MATLAB-like determination of occurence of assertion in an array using the
    numpy nonzero function
    """
    if not isinstance(assertions,tuple):
        raise ValueError("Input to 'find' needs to be a tuple.")
        
    inds = np.arange(len(assertions[0]))
    for assertion in assertions:
        inds = inds[np.nonzero(assertion[inds])[0]]
    
    return inds
            
def zerom(shape):
    """
    Generate a numpy.matrix of zeros of given shape
    """      
    from numpy import zeros, matrix      
    return matrix(zeros(shape))
    
    
def stack(elements):
    def make_matrix(v):
        if len(v.shape())>1:
            return v
        else:
            return v[:,np.newaxis]
            
    return np.hstack(map(lambda x: make_matrix(x), elements))
    

def print_vec(vector,prec=5,retS=False,vertical=False):
    if vertical:
        t = "\n"
    else:
        t = "\t"
    s = "".join(["%1.*g %s" % (int(prec),s,t) for s in vector])
    if retS:
        return s
    else:
        print(s)
        
def print_mat(matrix,prec=5,vertical=False,retS=False):
    if vertical:
        matrix = matrix.T
        
    s = "".join([ print_vec(matrix[k,:],prec=prec,vertical=False,retS=True)+"\n" for k in range(matrix.shape[0])])
    
    if retS:
        return s
    else:
        print(s)


def make_semiposdef(matrix,maxiter=10,tol=1e-12):

    n,m = matrix.shape
    if n!=m:
        raise ValueError("Matrix has to be quadratic")
    matrix = 0.5*(matrix+matrix.T)
    e = np.real(np.linalg.eigvals(matrix)).min()
    count = 0
    while e<tol and count<maxiter:
        matrix += (np.abs(e)+tol)*np.eye(n)
        e = np.real(np.linalg.eigvals(matrix)).min()
        count += 1
    return matrix


def FreqResp2RealImag(Abs, Phase, Unc, MCruns=1e4):
    """
    Calculation of real and imaginary parts from amplitude and phase with associated
    uncertainties.

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

    AbsPhas = np.random.multivariate_normal(np.hstack((Abs, Phase)), Unc, int(MCruns))

    H = AbsPhas[:, :Nf] * np.exp(1j * AbsPhas[:, Nf:])
    RI = np.hstack((np.real(H), np.imag(H)))

    Re = np.mean(RI[:, :Nf])
    Im = np.mean(RI[:, Nf:])
    URI = np.cov(RI, rowvar=False)

    return Re, Im, URI


def mapinside(a):
    """
    Mapping roots of the polynomial with coefficents a into the unit circle by projection.

    Parameters
    ----------
        a: ndarray of shape N - coefficients of polynomial 

    Returns
    -------
        v: ndarray of shape N - coefficients of polynomial with all roots inside the unit circle

    """
    from numpy import roots, conj, poly, nonzero
    v = roots(a)
    inds = nonzero(abs(v) > 1)
    v[inds] = 1 / conj(v[inds])
    return poly(v)