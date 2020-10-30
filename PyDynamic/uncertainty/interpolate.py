# -*- coding: utf-8 -*-
"""
The :mod:`PyDynamic.uncertainty.interpolate` module implements methods for
the propagation of uncertainties in the application of standard interpolation methods
as provided by :class:`scipy.interpolate.interp1d`.

This module contains the following function:

* :func:`interp1d_unc`: Interpolate arbitrary time series considering the associated
  uncertainties
"""
from typing import Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

__all__ = ["interp1d_unc", "make_equidistant"]


def interp1d_unc(
    x_new: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    uy: np.ndarray,
    kind: Optional[str] = "linear",
    copy=True,
    bounds_error: Optional[bool] = None,
    fill_value: Optional[Union[float, Tuple[float, float], str]] = np.nan,
    fill_unc: Optional[Union[float, Tuple[float, float], str]] = np.nan,
    assume_sorted: Optional[bool] = True,
    returnC: Optional[bool] = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    r"""Interpolate a 1-D function considering the associated uncertainties

    x and y are arrays of values used to approximate some function :math:`f \colon y
    = f(x)`.

    Note that calling :func:`interp1d_unc` with NaNs present in input values
    results in undefined behaviour.

    An equal number of each of the original x and y values and
    associated uncertainties is required.

    Parameters
    ----------
        x_new : (M,) array_like
            A 1-D array of real values to evaluate the interpolant at. x_new can be
            sorted in any order.
        x : (N,) array_like
            A 1-D array of real values.
        y : (N,) array_like
            A 1-D array of real values. The length of y must be equal to the length
            of x.
        uy : (N,) array_like
            A 1-D array of real values representing the standard uncertainties
            associated with y.
        kind : str, optional
            Specifies the kind of interpolation for y as a string ('previous',
            'next', 'nearest' or 'linear'). Default is ‘linear’.
        copy : bool, optional
            If True, the method makes internal copies of x and y. If False,
            references to x and y are used. The default is to copy.
        bounds_error : bool, optional
            If True, a ValueError is raised any time interpolation is attempted on a
            value outside of the range of x (where extrapolation is necessary). If
            False, out of bounds values are assigned fill_value. By default, an error
            is raised unless `fill_value="extrapolate"`.
        fill_value : array-like or (array-like, array_like) or “extrapolate”, optional

            - if or float, this value will be used to fill in for requested points
            outside of the data range. If not provided, then the default is NaN.
            - If a two-element tuple, then the first element is used as a fill value
              for `x_new < t[0]` and the second element is used for `x_new > t[-1]`.
              Anything that is not a 2-element tuple (e.g., list or ndarray, regardless
              of shape) is taken to be a single array-like argument meant to be used
              for both bounds as `below, above = fill_value, fill_value`.
            - If “extrapolate”, then points outside the data range will be set
              to the first or last element of the values.

            Both parameters `fill_value` and `fill_unc` should be
            provided to ensure desired behaviour in the extrapolation range.

        fill_unc : array-like or (array-like, array_like) or “extrapolate”, optional
            Usage and behaviour as described in `fill_value` but for the
            uncertainties. Both parameters `fill_value` and `fill_unc` should be
            provided to ensure desired behaviour in the extrapolation range.
        assume_sorted : bool, optional
            If False, values of x can be in any order and they are sorted first. If
            True, x has to be an array of monotonically increasing values.
        returnC : bool, optional
            If True, return sensitivity coefficients for later use. This is only
            available for interpolation kind 'linear' and for
            fill_unc="extrapolate" at the moment. If False sensitivity
            coefficients are not returned and internal computation is
            slightly more efficient.

    If `returnC` is False, which is the default behaviour, the method returns:

    Returns
    -------
        x_new : (M,) array_like
            values at which the interpolant is evaluated
        y_new : (M,) array_like
            interpolated values
        uy_new : (M,) array_like
            interpolated associated standard uncertainties

    Otherwise the method returns:

    Returns
    -------
        x_new : (M,) array_like
            values at which the interpolant is evaluated
        y_new : (M,) array_like
            interpolated values
        uy_new : (M,) array_like
            interpolated associated standard uncertainties
        C : (M,N) array_like
            sensitivity matrix :math:`C`, which is used to compute the uncertainties
            :math:`U_{y_{new}} = C \cdot \operatorname{diag}(u_y^2) \cdot C^T`

    References
    ----------
        * White [White2017]_
    """
    # This is taken from the class scipy.interpolate.interp1d to copy and sort the
    # arrays in case that is requested and of course extended by the uncertainties.
    # ----------------------------------------------------------------------------------
    x = np.array(x, copy=copy)
    y = np.array(y, copy=copy)
    uy = np.array(uy, copy=copy)

    if not assume_sorted:
        ind = np.argsort(x)
        x = x[ind]
        y = np.take(y, ind)
        uy = np.take(uy, ind)
    # ----------------------------------------------------------------------------------
    # Check for proper dimensions of inputs which are not checked as desired by SciPy.
    if not len(y) == len(uy):
        raise ValueError(
            "interp1d_unc: Array of associated measurement values' uncertainties are "
            "expected to be of the same length as the array of measurement values, "
            f"but we have len(y) = {len(y)} and len(uy) = {len(uy)}. Please "
            f"provide an array of {len(y)} standard uncertainties."
        )

    # Set up parameter dicts for calls of interp1d. We use it for interpolating the
    # values and "interpolation" (i.e. look-ups) of uncertainties in the trivial cases.
    interp1d_params = {
        "kind": kind,
        "copy": False,
        "bounds_error": bounds_error,
        "assume_sorted": True,
    }

    # Ensure constant values outside the original bounds by setting fill_value
    # and fill_unc accordingly but do not bother in case exception would be thrown.
    if not bounds_error:
        # For the extrapolation to rely on SciPy's handling of fill values in all
        # but the linear case we set those explicitly to boundary values of y and uy
        # respectively but handle those cases separately.
        if fill_value == "extrapolate":
            fill_value = y[0], y[-1]

        if fill_unc == "extrapolate":
            fill_unc = uy[0], uy[-1]
        elif bounds_error is not None and returnC:
            # This means bounds_error is intentionally set to False and we want to
            # extrapolate uncertainties with custom values. Additionally the sensitivity
            # coefficients shall be returned. This is not yet possible, because in this
            # case, we do not know, how to map the provided extrapolation values onto
            # the original values and thus we cannot provide the coefficients. Once we
            # deal with this, we will probably introduce another input parameter
            # fill_sens which is expected to be of shape (N,) or a 2-tuple of this
            # shape, which is then used in C wherever an extrapolation is performed.
            raise NotImplementedError(
                "interp1d_unc: This feature is not yet implemented.  We are planning "
                "to add another input parameter which is meant to carry the "
                "sensitivities for the extrapolated uncertainties. Get in touch with "
                "us, if you need it to discuss how to proceed."
            )

    # Inter- and extrapolate values in the desired fashion relying on SciPy.
    interp_y = interp1d(x, y, fill_value=fill_value, **interp1d_params)
    y_new = interp_y(x_new)

    if kind in ("previous", "next", "nearest"):
        if returnC:
            raise NotImplementedError(
                "interp1d_unc: Returning the sensitivity matrix for now is only "
                "supported for interpolation types other than 'previous', 'next' and "
                "'nearest'. Get in touch with us, if you need this to discuss how to "
                "proceed."
            )
        # Look up uncertainties.
        interp_uy = interp1d(x, uy, fill_value=fill_unc, **interp1d_params)
        uy_new = interp_uy(x_new)
    elif kind == "linear":
        # Calculate boolean arrays of indices from t_new which are outside t's bounds...
        extrap_range_below = x_new < np.min(x)
        extrap_range_above = x_new > np.max(x)
        extrap_range = extrap_range_below | extrap_range_above
        # .. and inside t's bounds.
        interp_range = ~extrap_range

        # Initialize the result array for the standard uncertainties.
        uy_new = np.empty_like(y_new)

        # Initialize the sensitivity matrix of shape (M, N) if needed.
        if returnC:
            C = np.zeros((len(x_new), len(uy)), "float64")

        # First extrapolate the according values if required and then
        # compute interpolated uncertainties following White, 2017.

        # If extrapolation is needed, fill in the values provided via fill_unc.
        if np.any(extrap_range):
            # At this point fill_unc is either a float (np.nan is a float as well) or
            # a 2-tuple of floats. In case we have one float we set uy_new to this value
            # inside the extrapolation range.
            if isinstance(fill_unc, float):
                uy_new[extrap_range] = fill_unc
            else:
                # Now fill_unc should be a 2-tuple, which we can fill into uy_new.
                uy_new[extrap_range_below], uy_new[extrap_range_above] = fill_unc

            if returnC:
                # In each row of C corresponding to an extrapolation value below the
                # original range set the first column to 1 and in each row of C
                # corresponding to an extrapolation value above the original range set
                # the last column to 1. It is important to do this before
                # interpolating, because in general those two columns can contain
                # non-zero values in the interpolation range.
                C[:, 0], C[:, -1] = extrap_range_below, extrap_range_above

        # If interpolation is needed, compute uncertainties following White, 2017.
        if np.any(interp_range):
            # This following section is taken mainly from scipy.interpolate.interp1d to
            # determine the indices of the relevant original x values just for the
            # interpolation range.
            # --------------------------------------------------------------------------
            # 2. Find where in the original data, the values to interpolate
            #    would be inserted.
            #    Note: If t_new[n] == t[m], then m is returned by searchsorted.
            t_new_indices = np.searchsorted(x, x_new[interp_range])

            # 3. Clip x_new_indices so that they are within the range of
            #    self.x indices and at least 1.  Removes mis-interpolation
            #    of x_new[n] = x[0]
            t_new_indices = t_new_indices.clip(1, len(x) - 1).astype(int)

            # 4. Calculate the slope of regions that each x_new value falls in.
            lo = t_new_indices - 1
            hi = t_new_indices

            t_lo = x[lo]
            t_hi = x[hi]
            # --------------------------------------------------------------------------
            if returnC:
                # Prepare the sensitivity coefficients, which in the first place
                # inside the interpolation range are the Lagrangian polynomials. We
                # compute the Lagrangian polynomials for all interpolation nodes
                # inside the original range.
                L_1 = (x_new[interp_range] - t_hi) / (t_lo - t_hi)
                L_2 = (x_new[interp_range] - t_lo) / (t_hi - t_lo)

                # Create iterators needed to efficiently fill our sensitivity matrix
                # in the rows corresponding to interpolation range.
                lo_it = iter(lo)
                hi_it = iter(hi)
                L_1_it = iter(L_1)
                L_2_it = iter(L_2)

                # In each row of C set the column with the corresponding
                # index in lo to L_1 and the column with the corresponding
                # index in hi to L_2.
                for index, C_row in enumerate(C):
                    if interp_range[index]:
                        C_row[next(lo_it)] = next(L_1_it)
                        C_row[next(hi_it)] = next(L_2_it)
                # Compute the standard uncertainties avoiding to build the sparse
                # covariance matrix diag(u_y^2). We reduce the equation C diag(u_y^2)
                # C^T for now to a more efficient calculation, which will work as
                # long as we deal with uncorrelated values, so that all information
                # can be found on the diagonal of the covariance and thus the result
                # matrix.
                uy_new[interp_range] = np.sqrt(
                    np.sum(C[interp_range] ** 2 * uy ** 2, 1)
                )
            else:
                # Since we do not need the sensitivity matrix, we compute uncertainties
                # more efficient (although we are actually not so sure about this
                # anymore). The simplification of the equation by pulling out the
                # denominator, just works because we work with the squared Lagrangians.
                # Otherwise we would have to account for the summation order.
                uy_prev_sqr = uy[lo] ** 2
                uy_next_sqr = uy[hi] ** 2
                uy_new[interp_range] = np.sqrt(
                    (x_new[interp_range] - t_hi) ** 2 * uy_prev_sqr
                    + (x_new[interp_range] - t_lo) ** 2 * uy_next_sqr
                ) / (t_hi - t_lo)
    else:
        raise NotImplementedError(
            f"interp1d_unc: The kind of interpolation '{kind}' is unsupported yet. Let "
            f"us know, that you need it."
        )

    if returnC:
        return x_new, y_new, uy_new, C
    return x_new, y_new, uy_new


def make_equidistant(
    x: np.ndarray,
    y: np.ndarray,
    uy: np.ndarray,
    dx: Optional[float] = 5e-2,
    kind: Optional[str] = "linear",
):
    r"""Interpolate a 1-D function equidistantly considering associated uncertainties

    Interpolate function values equidistantly and propagate uncertainties
    accordingly.

    x and y are arrays of values used to approximate some function :math:`f \colon y
    = f(x)`.

    Note that calling :func:`interp1d_unc` with NaNs present in input values
    results in undefined behaviour.

    An equal number of each of the original x and y values and
    associated uncertainties is required.

    Parameters
    ----------
        x: (N,) array_like
            A 1-D array of real values.
        y: (N,) array_like
            A 1-D array of real values. The length of y must be equal to the length
            of x.
        uy: (N,) array_like
            A 1-D array of real values representing the standard uncertainties
            associated with y.
        dx: float, optional
            desired interval length (defaults to 5e-2)
        kind : str, optional
            Specifies the kind of interpolation for y as a string ('previous',
            'next', 'nearest' or 'linear'). Default is ‘linear’.

    Returns
    -------
        x_new : (M,) array_like
            values at which the interpolant is evaluated
        y_new : (M,) array_like
            interpolated values
        uy_new : (M,) array_like
            interpolated associated standard uncertainties

    References
    ----------
        * White [White2017]_
    """
    # Find x's maximum.
    x_max = np.max(x)

    # Setup new vector of x values.
    x_new = np.arange(np.min(x), x_max, dx)

    # Since np.arange in overflow situations results in the biggest values not
    # guaranteed to be smaller than x's maximum', we need to check for this and delete
    # these unexpected values.
    if x_new[-1] > x_max:
        x_new = x_new[x_new <= x_max]

    return interp1d_unc(x_new, x, y, uy, kind)
