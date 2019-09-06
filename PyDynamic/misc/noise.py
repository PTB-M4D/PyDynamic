import numpy as np

__all__ = ["ARMA"]


def ARMA(length, phi=0.0, theta=0.0, std=1.0):
    r"""
    Generate time-series of a predefined ARMA-process based on this equation:
    :math:`\sum_{j=1}^{\min(p,n-1)} \phi_j \epsilon[n-j] + \sum_{j=1}^{\min(q,n-1)} \theta_j w[n-j]`
    where w is white gaussian noise. Equation and algorithm taken from [Eichst2012]_ .

    Parameters
    ----------
    length: int
        how long the drawn sample will be
    phi: float, list or numpy.ndarray, shape (p, )
        AR-coefficients
    theta: float, list or numpy.ndarray
        MA-coefficients
    std: float
        std of the gaussian white noise that is feeded into the ARMA-model

    Returns
    -------
    e: numpy.ndarray, shape (length, )
       time-series of the predefined ARMA-process

    References
    ----------
        * Eichstädt, Link, Harris and Elster [Eichst2012]_
    """

    # convert to numpy.ndarray
    if isinstance(phi, float):
        phi = np.array([phi])
    elif isinstance(phi, list):
        phi = np.array(phi)

    if isinstance(theta, float):
        theta = np.array([theta])
    elif isinstance(theta, list):
        theta = np.array(theta)

    # initialize e, w
    w = np.random.normal(loc=0, scale=std, size=length)
    e = np.zeros_like(w)

    # define shortcuts
    p = len(phi)
    q = len(theta)

    # iterate series over time
    for n, wn in enumerate(w):
        min_pn = min(p, n)
        min_qn = min(q, n)
        e[n] = np.sum(phi[:min_pn].dot(e[n-min_pn:n])) + np.sum(theta[:min_qn].dot(w[n-min_qn:n])) + wn

    return e
