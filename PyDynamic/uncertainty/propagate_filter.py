# -*- coding: utf-8 -*-
"""
This module contains functions for the propagation of uncertainties through
the application of a digital filter using the GUM approach.

This modules contains the following functions:

* :func:`FIRuncFilter`: Uncertainty propagation for signal y and uncertain FIR
  filter theta
* :func:`IIRuncFilter`: Uncertainty propagation for the signal x and the uncertain
  IIR filter (b,a)

.. note:: The Elster-Link paper for FIR filters assumes that the autocovariance
          is known and that noise is stationary!

"""

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import lfilter, lfilter_zi, dimpulse
from scipy.signal import convolve
from ..misc.tools import trimOrPad

__all__ = ["FIRuncFilter", "IIRuncFilter"]


def _fir_filter(x, theta, Ux=None, Utheta=None, initial_conditions="constant"):
    """Uncertainty propagation for signal x with covariance Ux
       and uncertain FIR filter theta with covariance Utheta.

       If either Ux or Utheta are omitted (None), then corresponding terms are not
       calculated to reduce computation time.

    Parameters
    ----------
    x : np.ndarray
        filter input signal
    theta : np.ndarray
        FIR filter coefficients
    Ux : np.ndarray, optional
        covariance matrix associated with x
        if the signal is fully certain, use `Ux = None` (default) to make use of more efficient calculations.
    Utheta : np.ndarray, optional
        covariance matrix associated with theta
        if the filter is fully certain, use `Utheta = None` (default) to make use of more efficient calculations.
        see also the comparison given in <examples\Digital filtering\FIRuncFilter_runtime_comparison.py>
    initial_conditions : str, optional
        constant: assume signal + uncertainty are constant before t=0 (default)
        zero: assume signal + uncertainty are zero before t=0


    Returns
    -------
    y : np.ndarray
        FIR filter output signal
    Uy : np.ndarray
        covariance matrix of filter output y


    References
    ----------
    * Elster and Link 2008 [Elster2008]_

    .. seealso:: :mod:`PyDynamic.model_estimation.fit_filter`

    """

    # Note to future developers:
    # The functions _fir_filter and _fir_filter_diag share
    # the same logic. If adjustments become necessary (e.g.
    # due to bug fixing) please also consider adjusting it
    # in the other function as well.

    Ntheta = len(theta)  # FIR filter size

    if initial_conditions == "constant":
        x0 = x[0]

    # Note: currently only used in testing for comparison against Monte Carlo method
    elif initial_conditions == "zero":
        x0 = 0.0

    else:
        raise ValueError(
            f"_fit_filter: You provided 'initial_conditions' = '{initial_conditions}'."
            f"However, only 'zero' or 'constant' are currently supported."
        )

    # propagate filter
    y, _ = lfilter(theta, 1.0, x, zi=x0 * lfilter_zi(theta, 1.0))

    # propagate uncertainty
    Uy = np.zeros((len(x), len(x)))

    ## only calculate subterms, that are non-zero (compare eq. 34 of paper)
    if Ux is not None:
        ## extend covariance Ntheta steps into the past
        if initial_conditions == "constant":
            Ux_extended = _stationary_prepend_covariance(Ux, Ntheta - 1)

        elif initial_conditions == "zero":
            # extend covariance Ntheta steps into the past
            Ux_extended = np.pad(
                Ux,
                ((Ntheta - 1, 0), (Ntheta - 1, 0)),
                "constant",
                constant_values=0,
            )

        # calc subterm theta^T * Ux * theta
        Uy += convolve(np.outer(theta, theta), Ux_extended, mode="valid")

    if Utheta is not None:
        ## extend signal Ntheta steps into the past
        x_extended = np.r_[np.full((Ntheta - 1), x0), x]

        # calc subterm x^T * Utheta * x
        Uy += convolve(np.outer(x_extended, x_extended), Utheta, mode="valid")

    if (Ux is not None) and (Utheta is not None):
        # calc subterm Tr(Ux * Utheta)
        Uy += convolve(Ux_extended, Utheta.T, mode="valid")

    return y, Uy


def _fir_filter_diag(
    x, theta, Ux_diag=None, Utheta_diag=None, initial_conditions="constant"
):
    """Uncertainty propagation for signal x with covariance diagonal Ux_diag
       and uncertain FIR filter theta with covariance diagonal Utheta_diag.

       If either Ux_diag or Utheta_diag are omitted (None), then corresponding terms are not
       calculated to reduce computation time.

    Parameters
    ----------
    x : np.ndarray
        filter input signal
    theta : np.ndarray
        FIR filter coefficients
    Ux_diag : np.ndarray, optional
        diagonal of covariance matrix ([ux11^2, ux22^2, ..., uxnn^2])
        if the signal is fully certain, use `Ux_diag = None` (default) to make use of more efficient calculations.
    Utheta_diag : np.ndarray, optional
        diagonal of covariance matrix ([ut11^2, ut22^2, ..., utnn^2])
        if the filter is fully certain, use `Utheta_diag = None` (default) to make use of more efficient calculations.
        see also the comparison given in <examples\Digital filtering\FIRuncFilter_runtime_comparison.py>
    initial_conditions : str, optional
        constant: assume signal + uncertainty are constant before t=0 (default)
        zero: assume signal + uncertainty are zero before t=0


    Returns
    -------
    y : np.ndarray
        FIR filter output signal
    Uy_diag : np.ndarray
        diagonal of covariance matrix of filter output y


    References
    ----------
    * Elster and Link 2008 [Elster2008]_

    .. seealso:: :mod:`PyDynamic.model_estimation.fit_filter`

    """

    # Note to future developers:
    # The functions _fir_filter and _fir_filter_diag share
    # the same logic. If adjustments become necessary (e.g.
    # due to bug fixing) please also consider adjusting it
    # in the other function as well.

    Ntheta = len(theta)  # FIR filter size

    if initial_conditions == "constant":
        x0 = x[0]

    # Note: currently only used in testing for comparison against Monte Carlo method
    elif initial_conditions == "zero":
        x0 = 0.0

    else:
        raise ValueError(
            f"_fit_filter: You provided 'initial_conditions' = '{initial_conditions}'."
            f"However, only 'zero' or 'constant' are currently supported."
        )

    # propagate filter
    y, _ = lfilter(theta, 1.0, x, zi=x0 * lfilter_zi(theta, 1.0))

    # propagate uncertainty
    Uy_diag = np.zeros(len(x))

    ## only calculate subterms, that are non-zero (compare eq. 34 of paper)
    if Ux_diag is not None:
        ## extend covariance Ntheta steps into the past
        if initial_conditions == "constant":
            Ux0 = Ux_diag[0]
        elif initial_conditions == "zero":
            Ux0 = 0.0
        Ux_diag_extended = np.r_[np.full((Ntheta - 1), Ux0), Ux_diag]

        # calc subterm theta^T * Ux * theta
        Uy_diag += convolve(np.square(theta), Ux_diag_extended, mode="valid")

    if Utheta_diag is not None:
        ## extend signal Ntheta steps into the past
        x_extended = np.r_[np.full((Ntheta - 1), x0), x]

        # calc subterm x^T * Utheta * x
        Uy_diag += convolve(np.square(x_extended), Utheta_diag, mode="valid")

    if (Ux_diag is not None) and (Utheta_diag is not None):
        # calc subterm Tr(Ux * Utheta)
        Uy_diag += convolve(Ux_diag_extended, Utheta_diag, mode="valid")

    return y, Uy_diag


def _stationary_prepend_covariance(U, n):
    """ Prepend covariance matrix U by n steps into the past"""

    c = np.r_[U[:, 0], np.zeros(n)]
    r = np.r_[U[0, :], np.zeros(n)]

    U_adjusted = toeplitz(c, r)
    U_adjusted[n:, n:] = U

    return U_adjusted


def FIRuncFilter(
    y,
    sigma_noise,
    theta,
    Utheta=None,
    shift=0,
    blow=None,
    kind="corr",
    return_full_covariance=False,
):
    """Uncertainty propagation for signal y and uncertain FIR filter theta

    A preceding FIR low-pass filter with coefficients `blow` can be provided optionally.

    This method keeps the signature of `PyDynamic.uncertainty.FIRuncFilter`, but internally
    works differently and can return a full covariance matrix. Also, sigma_noise can be a full
    covariance matrix.

    Parameters
    ----------
    y : np.ndarray
        filter input signal
    sigma_noise : float or np.ndarray
        float:    standard deviation of white noise in y
        1D-array: interpretation depends on kind
        2D-array: full covariance of input
    theta : np.ndarray
        FIR filter coefficients
    Utheta : np.ndarray, optional
        1D-array: coefficient-wise standard uncertainties of filter
        2D-array: covariance matrix associated with theta
        if the filter is fully certain, use `Utheta = None` (default) to make use of more efficient calculations.
        see also the comparison given in <examples\Digital filtering\FIRuncFilter_runtime_comparison.py>
    shift : int, optional
        time delay of filter output signal (in samples) (defaults to 0)
    blow : np.ndarray, optional
        optional FIR low-pass filter
    kind : string
        only meaningful in combination with sigma_noise a 1D numpy array
        "diag": point-wise standard uncertainties of non-stationary white noise
        "corr": single sided autocovariance of stationary (colored/correlated)
        noise (default)
    return_full_covariance : bool, optional
        whether or not to return a full covariance of the output, defaults to False

    Returns
    -------
    x : np.ndarray
        FIR filter output signal
    Ux : np.ndarray
        return_full_covariance == False : point-wise standard uncertainties associated with x (default)
        return_full_covariance == True : covariance matrix containing uncertainties associated with x


    References
    ----------
    * Elster and Link 2008 [Elster2008]_

    .. seealso:: :mod:`PyDynamic.model_estimation.fit_filter`

    """

    # Check for special cases, that we can compute much faster:
    ## These special cases come from the default behavior of the `old` FIRuncFilter
    ## implementation, which always returned only the square-root of the main diagonal
    ## of the covariance matrix.
    ## The special case is activated, if the user does not want a full covariance of
    ## the result (return_full_covariance == False). Furthermore, sigma_noise and Utheta
    ## must be representable by pure diagonal covariance matricies (i.e. they are of type
    ## None, float or 1D-array). By the same reasoning, we need to exclude cases
    ## of preceding low-pass filtering, as the covariance of the low-pass-filtered
    ## signal would no longer be purely diagonal.
    ## Computation is then based on eq. 33 [Elster2008]_ - which is substantially
    ## faster (typically by orders of magnitude) when compared to the full covariance
    ## calculation.
    ## Check this example: <examples\Digital filtering\FIRuncFilter_runtime_comparison.py>
    
    # note to user
    if not return_full_covariance:
        print(
            "FIRuncFilter: Output uncertainty will be given as 1D-array of point-wise "
            "standard uncertainties. Although this requires significantly lesser computations, "
            "it ignores correlation information. Every FIR-filtered signal will have "
            "off-diagonal entries in its covariance matrix (assuming the filter is longer "
            "than 1). To get the full output covariance matrix, use 'return_full_covariance=True'."
        )

    if (
        (not return_full_covariance)  # no need for full covariance
        and (  # cases of sigma_noise, that support the faster computation
            isinstance(sigma_noise, float)
            or (isinstance(sigma_noise, np.ndarray) and len(sigma_noise.shape) == 1)
            or (sigma_noise is None)
        )
        and (  # cases of Utheta, that support the faster computation
            isinstance(Utheta, float)
            or (isinstance(Utheta, np.ndarray) and len(Utheta.shape) == 1)
            or (Utheta is None)
        )
        # the low-pass-filtered signal wouldn't have pure diagonal covariance
        and (blow is None)
        # if sigma_noise is 1D, it must represent the diagonal of a covariance matrix
        and (kind == "diag" or not isinstance(sigma_noise, np.ndarray))
    ):

        if isinstance(sigma_noise, float):
            Uy_diag = np.full_like(y, sigma_noise ** 2)
        elif isinstance(sigma_noise, np.ndarray):
            Uy_diag = np.square(sigma_noise)
        else:
            Uy_diag = sigma_noise

        if isinstance(Utheta, float):
            Utheta_diag = np.full_like(theta, Utheta)
        elif isinstance(Utheta, np.ndarray):
            Utheta_diag = np.square(Utheta)
        else:
            Utheta_diag = Utheta

        x, Ux_diag = _fir_filter_diag(
            y, theta, Uy_diag, Utheta_diag, initial_conditions="constant"
        )
        return x, np.sqrt(np.abs(Ux_diag))

    # otherwise, the method computes full covariance information
    else:
        Ntheta = len(theta)  # FIR filter size

        # check which case of sigma_noise is necessary
        if isinstance(sigma_noise, float):
            Uy = np.diag(np.full(len(y), sigma_noise ** 2))

        elif isinstance(sigma_noise, np.ndarray):

            if len(sigma_noise.shape) == 1:
                if kind == "diag":
                    Uy = np.diag(sigma_noise ** 2)
                elif kind == "corr":
                    Uy = toeplitz(trimOrPad(sigma_noise, len(y)))
                else:
                    raise ValueError("unknown kind of sigma_noise")

            elif len(sigma_noise.shape) == 2:
                Uy = sigma_noise

        else:
            raise ValueError(
                "Unsupported value of sigma_noise. Please check the documentation."
            )

        # filter operation(s)
        if isinstance(blow, np.ndarray):
            # apply (fully certain) lowpass-filter
            xlow, Ulow = _fir_filter(y, blow, Uy, None, initial_conditions="constant")

            # apply filter to lowpass-filtered signal
            x, Ux = _fir_filter(
                xlow, theta, Ulow, Utheta, initial_conditions="constant"
            )

        else:
            # apply filter to input signal
            x, Ux = _fir_filter(y, theta, Uy, Utheta, initial_conditions="constant")

        # shift result
        if shift != 0:
            x = np.roll(x, -int(shift))
            Ux = np.roll(Ux, (-int(shift), -int(shift)))

        if return_full_covariance:
            return x, Ux
        else:
            return x, np.sqrt(np.abs(np.diag(Ux)))


def IIRuncFilter(x, noise, b, a, Uab):
    """
    Uncertainty propagation for the signal x and the uncertain IIR filter (b,a)

    Parameters
    ----------
    x: np.ndarray
        filter input signal
    noise: float
        signal noise standard deviation
    b: np.ndarray
        filter numerator coefficients
    a: np.ndarray
        filter denominator coefficients
    Uab: np.ndarray
        covariance matrix for (a[1:],b)

    Returns
    -------
    y: np.ndarray
        filter output signal
    Uy: np.ndarray
        uncertainty associated with y

    References
    ----------
        * Link and Elster [Link2009]_

    """

    if not isinstance(noise, np.ndarray):
        noise = noise * np.ones_like(x)  # translate iid noise to vector

    p = len(a) - 1

    # Adjust dimension for later use.
    if not len(b) == len(a):
        b = np.hstack((b, np.zeros((len(a) - len(b),))))

    # From discrete-time transfer function to state space representation.
    [A, bs, c, b0] = tf2ss(b, a)

    A = np.matrix(A)
    bs = np.matrix(bs)
    c = np.matrix(c)

    phi = np.zeros((2 * p + 1, 1))
    dz = np.zeros((p, p))
    dz1 = np.zeros((p, p))
    z = np.zeros((p, 1))
    P = np.zeros((p, p))

    y = np.zeros((len(x),))
    Uy = np.zeros((len(x),))

    Aabl = np.zeros((p, p, p))
    for k in range(p):
        Aabl[0, k, k] = -1

    # implementation of the state-space formulas from the paper
    for n in range(len(y)):
        for k in range(p):  # derivative w.r.t. a_1,...,a_p
            dz1[:, k] = A * dz[:, k] + np.squeeze(Aabl[:, :, k]) * z
            phi[k] = c * dz[:, k] - b0 * z[k]
        phi[p + 1] = -np.matrix(a[1:]) * z + x[n]  # derivative w.r.t. b_0
        for k in range(p + 2, 2 * p + 1):  # derivative w.r.t. b_1,...,b_p
            phi[k] = z[k - (p + 1)]
        P = A * P * A.T + noise[n] ** 2 * (bs * bs.T)
        y[n] = c * z + b0 * x[n]
        Uy[n] = phi.T * Uab * phi + c * P * c.T + b[0] ** 2 * noise[n] ** 2
        z = A * z + bs * x[n]  # update of the state equations
        dz = dz1

    Uy = np.sqrt(np.abs(Uy))  # calculate point-wise standard uncertainties

    return y, Uy
