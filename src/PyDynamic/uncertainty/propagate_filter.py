"""
This module contains functions for the propagation of uncertainties through
the application of a digital filter using the GUM approach.

This modules contains the following functions:

* :func:`FIRuncFilter`: Uncertainty propagation for signal y and uncertain FIR
  filter theta
* :func:`IIRuncFilter`: Uncertainty propagation for the signal x and the uncertain
  IIR filter (b,a)
* :func:`IIR_get_initial_state`: Get a valid internal state for :func:`IIRuncFilter`
  that assumes a stationary signal before the first value.

.. note:: The Elster-Link paper for FIR filters assumes that the autocovariance
          is known and that noise is stationary!

"""
import warnings

import numpy as np
from scipy.linalg import solve, solve_discrete_lyapunov, toeplitz
from scipy.signal import convolve, dimpulse, lfilter, lfilter_zi

from ..misc.tools import trimOrPad

__all__ = ["FIRuncFilter", "IIRuncFilter", "IIR_get_initial_state"]


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
        covariance matrix associated with theta. If the filter is fully certain,
        do not provide Utheta to make use of more efficient calculations.
        see also the comparison given in <examples\Digital filtering\FIRuncFilter_runtime_comparison.py>
    initial_conditions : str, optional
        - "constant": assume signal + uncertainty are constant before t=0 (default)
        - "zero": assume signal + uncertainty are zero before t=0


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
        Uy += _clip_main_diagonal_to_zero_from_below(
            convolve(np.outer(theta, theta), Ux_extended, mode="valid")
        )

    if Utheta is not None:
        ## extend signal Ntheta steps into the past
        x_extended = np.r_[np.full((Ntheta - 1), x0), x]

        # calc subterm x^T * Utheta * x
        Uy += _clip_main_diagonal_to_zero_from_below(
            convolve(np.outer(x_extended, x_extended), Utheta, mode="valid")
        )

    if (Ux is not None) and (Utheta is not None):
        # calc subterm Tr(Ux * Utheta)
        Uy += _clip_main_diagonal_to_zero_from_below(
            convolve(Ux_extended, Utheta.T, mode="valid")
        )

    return y, Uy


def _clip_main_diagonal_to_zero_from_below(matrix: np.ndarray) -> np.ndarray:
    np.fill_diagonal(matrix, matrix.diagonal().clip(min=0))
    return matrix


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
        Uy_diag += convolve(np.square(theta), Ux_diag_extended, mode="valid").clip(
            min=0
        )

    if Utheta_diag is not None:
        ## extend signal Ntheta steps into the past
        x_extended = np.r_[np.full((Ntheta - 1), x0), x]

        # calc subterm x^T * Utheta * x
        Uy_diag += convolve(np.square(x_extended), Utheta_diag, mode="valid").clip(
            min=0
        )

    if (Ux_diag is not None) and (Utheta_diag is not None):
        # calc subterm Tr(Ux * Utheta)
        Uy_diag += convolve(Ux_diag_extended, Utheta_diag, mode="valid").clip(min=0)

    return y, Uy_diag


def _stationary_prepend_covariance(U, n):
    """Prepend covariance matrix U by n steps into the past"""

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
        - float: standard deviation of white noise in y
        - 1D-array: interpretation depends on kind
        - 2D-array: full covariance of input

    theta : np.ndarray
        FIR filter coefficients
    Utheta : np.ndarray, optional
        - 1D-array: coefficient-wise standard uncertainties of filter
        - 2D-array: covariance matrix associated with theta

        if the filter is fully certain, use `Utheta = None` (default) to make use of more efficient calculations.
        see also the comparison given in <examples\Digital filtering\FIRuncFilter_runtime_comparison.py>
    shift : int, optional
        time delay of filter output signal (in samples) (defaults to 0)
    blow : np.ndarray, optional
        optional FIR low-pass filter
    kind : string, optional
        only meaningful in combination with sigma_noise a 1D numpy array

        - "diag": point-wise standard uncertainties of non-stationary white noise
        - "corr": single sided autocovariance of stationary colored noise (default)

    return_full_covariance : bool, optional
        whether or not to return a full covariance of the output, defaults to False

    Returns
    -------
    x : np.ndarray
        FIR filter output signal
    Ux : np.ndarray
        - return_full_covariance == False : point-wise standard uncertainties
          associated with x (default)
        - return_full_covariance == True : covariance matrix containing uncertainties
          associated with x

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
    ## Check this example: <examples\digital_filtering\FIRuncFilter_runtime_comparison.py>

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

        Ux_diag = np.sqrt(np.abs(Ux_diag))

        # shift result
        if shift != 0:
            x = np.roll(x, -int(shift))
            Ux_diag = np.roll(Ux_diag, -int(shift))

        return x, Ux_diag

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
                    raise ValueError(
                        f"Unknown kind `{kind}`. Don't now how to interpret the array "
                        f"sigma_noise."
                    )

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
            Ux = np.roll(Ux, (-int(shift), -int(shift)), axis=(0, 1))

        if return_full_covariance:
            return x, Ux
        else:
            return x, np.sqrt(np.abs(np.diag(Ux)))


def IIRuncFilter(x, Ux, b, a, Uab=None, state=None, kind="corr"):
    """
    Uncertainty propagation for the signal x and the uncertain IIR filter (b,a)

    Parameters
    ----------
        x : np.ndarray
            filter input signal
        Ux : float or np.ndarray
            float:    standard deviation of white noise in x (requires `kind="diag"`)
            1D-array: interpretation depends on kind
        b : np.ndarray
            filter numerator coefficients
        a : np.ndarray
            filter denominator coefficients
        Uab : np.ndarray, optional (default: None)
            covariance matrix for (a[1:],b)
        state : dict, optional (default: None)
            An internal state (z, dz, P, cache) to start from - e.g. from a previous run of IIRuncFilter.

            * If not given, (z, dz, P) are calculated such that the signal was constant before the given range
            * If given, the input parameters (b, a, Uab) are ignored to avoid repetitive rebuild of the
              internal system description (instead, the cache is used). However a valid new state (i.e.
              with new b, a, Uab) can always be generated by using :func:`IIR_get_initial_state`.
        kind : string, optional (default: "corr")
            defines the interpretation of Ux, if Ux is a 1D-array
            "diag": point-wise standard uncertainties of non-stationary white noise
            "corr": single sided autocovariance of stationary (colored/correlated) noise (default)

    Returns
    -------
        y : np.ndarray
            filter output signal
        Uy : np.ndarray
            uncertainty associated with y
        state : dict
            dictionary of updated internal state

    Note
    ----
        In case of `a == [1.0]` (FIR filter), the results of :func:`IIRuncFilter` and :func:`FIRuncFilter` might differ!
        This is because IIRuncFilter propagates uncertainty according to the
        (first-order Taylor series of the) GUM, whereas FIRuncFilter takes full
        variance information into account (which leads to an additional term).
        This is documented in the description of formula (33) of [Elster2008]_ .
        The difference can be visualized by running `PyDynamic/examples/digital_filtering/validate_FIR_IIR_MC.py`

    References
    ----------
        * Link and Elster [Link2009]_

    """

    # check user input
    if kind not in ("diag", "corr"):
        raise ValueError(
            "`kind` is expected to be either 'diag' or 'corr' but '{KIND}' was given.".format(
                KIND=kind
            )
        )

    # make Ux an array
    if not isinstance(Ux, np.ndarray):
        Ux = np.full(x.shape, Ux)  # translate iid noise to vector

        if kind != "diag":
            kind = "diag"
            warnings.warn(
                f"Ux of type float and `kind='{kind}'` was given. To ensure the "
                f"behavior described in the docstring (float -> standard deviation of "
                f"white noise  in x), `kind='diag'` is set. \n("
                "To suppress this warning, explicitly set `kind='diag'`",
                category=UserWarning,
            )

    # system, corr_unc and processed_input are cached as well to reduce computational load
    if state is None:
        # calculate initial state
        if kind == "diag":
            state = IIR_get_initial_state(b, a, Uab=Uab, x0=x[0], U0=Ux[0])
        else:  # "corr"
            state = IIR_get_initial_state(
                b, a, Uab=Uab, x0=x[0], U0=np.sqrt(Ux[0]), Ux=Ux
            )

    z = state["z"]
    dz = state["dz"]
    P = state["P"]
    A, bs, cT, b0 = state["cache"]["system"]
    corr_unc = state["cache"]["corr_unc"]
    b, a, Uab = state["cache"]["processed_input"]
    p = len(a) - 1

    # phi: dy/dtheta
    phi = np.empty((2 * p + 1, 1))

    # output y, output uncertainty Uy
    y = np.zeros_like(x)
    Uy = np.zeros_like(x)

    # implementation of the state-space formulas from the paper
    for n in range(len(y)):

        # calculate output according to formulas (7)
        y[n] = cT @ z + b0 * x[n]  # (7)

        # if Uab is not given, use faster implementation
        if isinstance(Uab, np.ndarray):
            # calculate phi according to formulas (13) and (15) from paper
            phi[:p] = np.transpose(
                cT @ dz - np.transpose(b0 * z[::-1])
            )  # derivative w.r.t. a_1,...,a_p
            phi[p] = -a[1:][::-1] @ z + x[n]  # derivative w.r.t. b_0
            phi[p + 1 :] = z[::-1]  # derivative w.r.t. b_1,...,b_p

            # output uncertainty according to formulas (12) and (19)
            if kind == "diag":
                Uy[n] = (
                    phi.T @ Uab @ phi + cT @ P @ cT.T + np.square(b0 * Ux[n])
                )  # (12)
            else:  # "corr"
                Uy[n] = phi.T @ Uab @ phi + corr_unc  # (19)

        else:  # Uab is None
            # output uncertainty according to formulas (12) and (19)
            if kind == "diag":
                Uy[n] = cT @ P @ cT.T + np.square(b0 * Ux[n])  # (12)
            else:  # "corr"
                Uy[n] = corr_unc  # (19)

        # timestep update preparations
        if kind == "diag":
            u_square = np.square(Ux[n])  # as in formula (18)
        else:  # "corr"
            u_square = Ux[0]  # adopted for kind == "corr"

        # | DON'T | # because dA is sparse, this is not efficient:
        # | USE   | # dA = _get_derivative_A(p)
        # | THIS  | # dA_z = np.hstack(dA @ z)
        # this is efficient, because no tensor-multiplication is involved:
        dA_z = np.vstack((np.zeros((p - 1, p)), -z[::-1].T))

        # timestep update
        P = A @ P @ A.T + u_square * np.outer(bs, bs)  # state uncertainty, formula (18)
        dz = A @ dz + dA_z  # state derivative, formula (17)
        z = A @ z + bs * x[n]  # state, formula (6)

    Uy = np.sqrt(np.abs(Uy))  # calculate point-wise standard uncertainties

    # return result and internal state
    state.update({"z": z, "dz": dz, "P": P})
    return y, Uy, state


def _tf2ss(b, a):
    """
    Variant of :func:`scipy.signal.tf2ss` that fits the definitions of [Link2009]_
    """

    p = len(a) - 1
    A = np.vstack([np.eye(p - 1, p, k=1), -a[1:][::-1]])
    B = np.zeros((p, 1))
    B[-1] = 1
    C = np.expand_dims((b[1:] - b[0] * a[1:])[::-1], axis=0)
    D = np.ones((1, 1)) * b[0]

    return A, B, C, D


def _get_derivative_A(size_A):
    """
    build tensor representing dA/dtheta
    """
    dA = np.zeros((size_A, size_A, size_A))
    for k in range(size_A):
        dA[k, -1, -(k + 1)] = -1

    return dA


def _get_corr_unc(b, a, Ux):
    """
    Calculate the cumulated correlated noise based on equations (20) of [Link2009]_ .
    """

    # get impulse response of IIR defined by (b,a)
    h_theta = np.squeeze(
        dimpulse((b, a, 1), x0=0.0, t=np.arange(0, len(Ux), step=1))[1][0]
    )

    # equation (20), note:
    # - for values r<0 or s<0 the contribution to the sum is zero (because h_theta is zero)
    # - Ux is the one-sided autocorrelation and assumed to be zero outside its range
    corr_unc = np.sum(toeplitz(Ux) * np.outer(h_theta, h_theta))

    return corr_unc


def IIR_get_initial_state(b, a, Uab=None, x0=1.0, U0=1.0, Ux=None):
    """
    Calculate the internal state for the IIRuncFilter-function corresponding to stationary
    non-zero input signal.

    Parameters
    ----------
        b : np.ndarray
            filter numerator coefficients
        a : np.ndarray
            filter denominator coefficients
        Uab : np.ndarray, optional (default: None)
            covariance matrix for (a[1:],b)
        x0 : float, optional (default: 1.0)
            stationary input value
        U0 : float, optional (default: 1.0)
            stationary input uncertainty
        Ux : np.ndarray, optional (default: None)
            single sided autocovariance of stationary (colored/correlated) noise
            (needed in the `kind="corr"` case of :func:`IIRuncFilter`)

    Returns
    -------
    internal_state : dict
        dictionary of state

    """

    # adjust filter coefficients for lengths
    b, a, Uab = _adjust_filter_coefficients(b, a, Uab)

    # convert into state space representation
    A, B, C, D = _tf2ss(b, a)

    # necessary intermediate variables
    p = len(A)
    IminusA = np.eye(p) - A
    dA = _get_derivative_A(p)

    # stationary internal state
    # (eye()-A) * zs = B*x0
    zs = solve(IminusA, B) * x0

    # stationary derivative of internal state
    # (eye() - A) dzs = dA * zs
    dzs = solve(IminusA, np.hstack(dA @ zs))

    # stationary uncertainty of internal state
    # A * Ps * A^T - Ps + u^2 * B * B^T = 0
    Ps = solve_discrete_lyapunov(A, U0 ** 2 * np.outer(B, B))

    if isinstance(Ux, np.ndarray):
        corr_unc = _get_corr_unc(b, a, Ux)
    else:
        corr_unc = 0

    # bring results into the format that is used within IIRuncFilter
    # this is the only place, where the structure of the cache is "documented"
    cache = {
        "system": (A, B, C, D),
        "corr_unc": corr_unc,
        "processed_input": (b, a, Uab),
    }
    state = {"z": zs, "dz": dzs, "P": Ps, "cache": cache}

    return state


def _adjust_filter_coefficients(b, a, Uab):
    """
    Bring b, a to the same length and adjust Uab accordingly
    """
    # length difference of both filters
    d = len(a) - len(b)

    # adjust filter coefficient uncertainties
    if isinstance(Uab, np.ndarray):
        if d != 0:
            l_theta = len(a) + len(b) - 1
            Uab_expected_shape = (l_theta, l_theta)
            if Uab.shape == Uab_expected_shape:
                if d < 0:
                    Uab = np.insert(
                        np.insert(Uab, [len(a) - 1] * (-d), 0, axis=0),
                        [len(a) - 1] * (-d),
                        0,
                        axis=1,
                    )
                elif d > 0:
                    Uab = np.hstack((Uab, np.zeros((Uab.shape[0], d))))
                    Uab = np.vstack((Uab, np.zeros((d, Uab.shape[1]))))
            else:
                raise ValueError(
                    "Uab is of shape {ACTUAL_SHAPE}, but expected shape is {EXPECTED_SHAPE} "
                    "(len(a)+len(b)-1)".format(
                        ACTUAL_SHAPE=Uab.shape, EXPECTED_SHAPE=Uab_expected_shape
                    )
                )

    # adjust filter coefficients for later use
    if d < 0:
        a = np.hstack((a, np.zeros(-d)))
    elif d > 0:
        b = np.hstack((b, np.zeros(d)))

    return b, a, Uab
