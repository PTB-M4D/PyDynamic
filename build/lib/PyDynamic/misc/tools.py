# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Sascha Eichstaedt (sascha.eichstaedt@ptb.de)
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