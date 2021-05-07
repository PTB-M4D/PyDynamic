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
from scipy.interpolate import interp1d, splrep, BSpline

__all__ = ["interp1d_unc", "make_equidistant"]


def interp1d_unc(
    t_new: np.ndarray,
    t: np.ndarray,
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

    t and y are arrays of values used to approximate some function :math:`f \colon y
    = f(t)`.

    Note that calling :func:`interp1d_unc` with NaNs present in input values
    results in undefined behaviour.

    An equal number of each of the original timestamps (or frequencies), values and
    associated uncertainties is required.

    Parameters
    ----------
        t_new : (M,) array_like
            A 1-D array of real values representing the timestamps (or frequencies) at
            which to evaluate the interpolated values. t_new can be sorted in any order.
        t : (N,) array_like
            A 1-D array of real values representing timestamps (or frequencies) in
            ascending order.
        y : (N,) array_like
            A 1-D array of real values. The length of y must be equal to the length
            of t.
        uy : (N,) array_like
            A 1-D array of real values representing the standard uncertainties
            associated with y.
        kind : str, optional
            Specifies the kind of interpolation for y as a string ('previous',
            'next', 'nearest', 'linear' or 'cubic'). Default is ‘linear’.
        copy : bool, optional
            If True, the method makes internal copies of t and y. If False,
            references to t and y are used. The default is to copy.
        bounds_error : bool, optional
            If True, a ValueError is raised any time interpolation is attempted on a
            value outside of the range of x (where extrapolation is necessary). If
            False, out of bounds values are assigned fill_value. By default, an error
            is raised unless `fill_value="extrapolate"`.
        fill_value : array-like or (array-like, array_like) or “extrapolate”, optional

            - if a ndarray (or float), this value will be used to fill in for
              requested points outside of the data range. If not provided, then the
              default is NaN.
            - If a two-element tuple, then the first element is used as a fill value
              for `t_new < t[0]` and the second element is used for `t_new > t[-1]`.
              Anything that is not a 2-element tuple (e.g., list or ndarray, regardless
              of shape) is taken to be a single array-like argument meant to be used
              for both bounds as `below, above = fill_value, fill_value`.
            - If “extrapolate”, then points outside the data range will be set
              to the first or last element of the values.
            - If cubic-interpolation, C2-continuity at the transition to the 
              extrapolation-range is not guaranteed. This behavior might change
              in future implementations, see issue #210 for details. 

            Both parameters `fill_value` and `fill_unc` should be
            provided to ensure desired behaviour in the extrapolation range.

        fill_unc : array-like or (array-like, array_like) or “extrapolate”, optional
            Usage and behaviour as described in `fill_value` but for the
            uncertainties. Both parameters `fill_value` and `fill_unc` should be
            provided to ensure desired behaviour in the extrapolation range.

        assume_sorted : bool, optional
            If False, values of t can be in any order and they are sorted first. If
            True, t has to be an array of monotonically increasing values.
        returnC : bool, optional
            If True, return sensitivity coefficients for later use. This is only
            available for interpolation kind 'linear' and for
            fill_unc="extrapolate" at the moment. If False sensitivity
            coefficients are not returned and internal computation is
            slightly more efficient.

    If `returnC` is False, which is the default behaviour, the method returns:

    Returns
    -------
        t_new : (M,) array_like
            interpolation timestamps (or frequencies)
        y_new : (M,) array_like
            interpolated values
        uy_new : (M,) array_like
            interpolated associated standard uncertainties

    Otherwise the method returns:

    Returns
    -------
        t_new : (M,) array_like
            interpolation timestamps (or frequencies)
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
    t = np.array(t, copy=copy)
    y = np.array(y, copy=copy)
    uy = np.array(uy, copy=copy)

    if not assume_sorted:
        ind = np.argsort(t)
        t = t[ind]
        y = np.take(y, ind)
        uy = np.take(uy, ind)
    # ----------------------------------------------------------------------------------
    # Check for proper dimensions of inputs which are not checked as desired by SciPy.
    if not len(y) == len(uy):
        raise ValueError(
            "Array of associated measurement values' uncertainties must be same length "
            "as array of measurement values."
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
                "This feature is not yet implemented.  We are planning to add "
                "another input parameter which is meant to carry the sensitivities "
                "for the extrapolated uncertainties. Get in touch with us, "
                "if you need it to discuss how to proceed."
            )

    # Inter- and extrapolate values in the desired fashion relying on SciPy.
    interp_y = interp1d(t, y, fill_value=fill_value, **interp1d_params)
    y_new = interp_y(t_new)

    if kind in ("previous", "next", "nearest"):
        if returnC:
            raise NotImplementedError(
                "Returning the sensitivity matrix for now is only supported for "
                "interpolation types other than 'previous', 'next' and 'nearest'. Get"
                "in touch with us, if you need this to discuss how to proceed."
            )
        # Look up uncertainties.
        interp_uy = interp1d(t, uy, fill_value=fill_unc, **interp1d_params)
        uy_new = interp_uy(t_new)
    elif kind in ("linear", "cubic"):
        # Calculate boolean arrays of indices from t_new which are outside t's bounds...
        extrap_range_below = t_new < np.min(t)
        extrap_range_above = t_new > np.max(t)
        extrap_range = extrap_range_below | extrap_range_above
        # .. and inside t's bounds.
        interp_range = ~extrap_range

        # Initialize the result array for the standard uncertainties.
        uy_new = np.empty_like(y_new)

        # Initialize the sensitivity matrix of shape (M, N) if needed.
        if returnC or kind == "cubic":
            C = np.zeros((len(t_new), len(uy)), "float64")

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

            if kind == "linear":
                # This following section is taken mainly from
                # scipy.interpolate.interp1d to determine the indices of the relevant
                # original timestamps (or frequencies) just for the interpolation range.
                # ----------------------------------------------------------------------
                # 2. Find where in the original data, the values to interpolate
                #    would be inserted.
                #    Note: If t_new[n] == t[m], then m is returned by searchsorted.
                t_new_indices = np.searchsorted(t, t_new[interp_range])

                # 3. Clip x_new_indices so that they are within the range of
                #    self.x indices and at least 1.  Removes mis-interpolation
                #    of x_new[n] = x[0]
                t_new_indices = t_new_indices.clip(1, len(t) - 1).astype(int)

                # 4. Calculate the slope of regions that each x_new value falls in.
                lo = t_new_indices - 1
                hi = t_new_indices

                t_lo = t[lo]
                t_hi = t[hi]
                # ----------------------------------------------------------------------
                if returnC:
                    # Prepare the sensitivity coefficients, which in the first place
                    # inside the interpolation range are the Lagrangian polynomials. We
                    # compute the Lagrangian polynomials for all interpolation nodes
                    # inside the original range.
                    L_1 = (t_new[interp_range] - t_hi) / (t_lo - t_hi)
                    L_2 = (t_new[interp_range] - t_lo) / (t_hi - t_lo)

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
                    # covariance matrix diag(u_y^2). We reduce the equation C diag(
                    # u_y^2) C^T for now to a more efficient calculation, which will
                    # work as long as we deal with uncorrelated values, so that all
                    # information can be found on the diagonal of the covariance and
                    # thus the result matrix.
                    uy_new[interp_range] = np.sqrt(
                        np.sum(C[interp_range] ** 2 * uy ** 2, 1)
                    )
                else:
                    # Since we do not need the sensitivity matrix, we compute
                    # uncertainties more efficient (although we are actually not so
                    # sure about this anymore). The simplification of the equation by
                    # pulling out the denominator, just works because we work with
                    # the squared Lagrangians. Otherwise we would have to account for
                    # the summation order.
                    uy_prev_sqr = uy[lo] ** 2
                    uy_next_sqr = uy[hi] ** 2
                    uy_new[interp_range] = np.sqrt(
                        (t_new[interp_range] - t_hi) ** 2 * uy_prev_sqr
                        + (t_new[interp_range] - t_lo) ** 2 * uy_next_sqr
                    ) / (t_hi - t_lo)

            elif kind == "cubic":
                # Calculate the uncertainty by generating a spline of sensitivity
                # coefficients. This procedure is described by eq. (19) of White2017.
                F_is = []
                for i in range(len(t)):
                    x_temp = np.zeros_like(t)
                    x_temp[i] = 1.0
                    F_i = BSpline(*splrep(t, x_temp, k=3))
                    F_is.append(F_i)

                # Calculate sensitivity coefficients.
                C[interp_range] = np.array([F_i(t_new[interp_range]) for F_i in F_is]).T
                C_sqr = np.square(C[interp_range])

                # if at some point time-uncertainties are of interest, White2017
                # already provides the formulas (eq. (17))

                # ut = np.zeros_like(t)
                # ut_new = np.zeros_like(t_new)
                # a1 = np.dot(C_sqr, np.square(uy))
                # a2 = np.dot(
                #     C_sqr,
                #     np.squeeze(np.square(interp_y._spline(t, nu=1))) * np.square(ut),
                # )
                # a3 = np.square(np.squeeze(interp_y._spline(t_new, nu=1))) * np.square(
                #     ut_new
                # )
                # uy_new[interp_range] = np.sqrt(a1 - a2 + a3)

                # without consideration of time-uncertainty eq. (17) becomes
                uy_new[interp_range] = np.sqrt(np.dot(C_sqr, np.square(uy)))
    else:
        raise NotImplementedError(
            "%s is unsupported yet. Let us know, that you need it." % kind
        )

    if returnC:
        return t_new, y_new, uy_new, C
    return t_new, y_new, uy_new


def make_equidistant(t, y, uy, dt=5e-2, kind="linear"):
    """Interpolate non-equidistant time series to equidistant

    Interpolate measurement values and propagate uncertainties accordingly.

    Parameters
    ----------
        t: (N,) array_like
            timestamps (or frequencies)
        y: (N,) array_like
            corresponding measurement values
        uy: (N,) array_like
            corresponding measurement values' standard uncertainties
        dt: float, optional
            desired interval length
        kind: str, optional
            Specifies the kind of interpolation for the measurement values
            as a string ('previous', 'next', 'nearest' or 'linear').

    Returns
    -------
        t_new : (M,) array_like
            interpolation timestamps (or frequencies)
        y_new : (M,) array_like
            interpolated measurement values
        uy_new : (M,) array_like
            interpolated measurement values' standard uncertainties

    References
    ----------
        * White [White2017]_
    """
    # Find t's maximum.
    t_max = np.max(t)

    # Setup new vector of timestamps.
    t_new = np.arange(np.min(t), t_max, dt)

    # Since np.arange in overflow situations results in the biggest values not
    # guaranteed to be smaller than t's maximum', we need to check for this and delete
    # these unexpected values.
    if t_new[-1] > t_max:
        t_new = t_new[t_new <= t_max]

    return interp1d_unc(t_new, t, y, uy, kind)
