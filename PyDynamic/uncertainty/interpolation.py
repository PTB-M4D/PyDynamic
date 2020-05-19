# -*- coding: utf-8 -*-
"""
The :mod:`PyDynamic.uncertainty.interpolation` module implements methods for
the propagation of uncertainties in the application of standard interpolation methods
as provided by :class:`scipy.interpolate.interp1d`.

This module contains the following function:

* :func:`interp1d_unc`: Interpolate arbitrary time series considering the associated
  uncertainties
"""
from typing import Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

__all__ = ["interp1d_unc"]


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
    return_c: Optional[bool] = False,
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
            'next', 'nearest' or 'linear'). Default is ‘linear’.
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

            Both parameters `fill_value` and `fill_unc` should be
            provided to ensure desired behaviour in the extrapolation range.

        fill_unc : array-like or (array-like, array_like) or “extrapolate”, optional
            Usage and behaviour as described in `fill_value` but for the
            uncertainties. Both parameters `fill_value` and `fill_unc` should be
            provided to ensure desired behaviour in the extrapolation range.

        assume_sorted : bool, optional
            If False, values of t can be in any order and they are sorted first. If
            True, t has to be an array of monotonically increasing values.
        return_c : bool, optional
            If True, return sensitivity coefficients for later use. This is only
            available for interpolation types other than 'previous', 'next' and
            'nearest'. If False sensitivity coefficients are not returned and
            internal computation is more efficient.

    If `return_c` is False, which is the default behaviour, the method returns:

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
            :math:`U_{y_{new}} = C \operatorname{diag}(u_y^2) C^T`

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

    # Set up parameter dicts for calls of interp1d. We use it for actual
    # "interpolation" (i.e. look-ups) in trivial cases and in the linear case to
    # extrapolate.
    interp1d_params = {
        "kind": kind,
        "copy": False,
        "bounds_error": bounds_error,
        "assume_sorted": True,
    }

    # Ensure constant behaviour outside the original bounds by setting fill_value
    # and fill_unc accordingly but do not bother in case exception would be thrown.
    if not bounds_error:
        # For the extrapolation to rely on SciPy's handling of fill values in all
        # but the linear case we set those explicitly to boundary values of y and uy
        # respectively.
        if fill_value == "extrapolate":
            fill_value = y[0], y[-1]
        if fill_unc == "extrapolate":
            fill_unc = uy[0], uy[-1]
        elif return_c:
            # This case is not so clear in terms of how to align the extrapolated
            # uncertainties with the associated uncertainties of the values,
            # so we throw an exception for now. We are not sure yet, how we should
            # extend the originally provided uncertainties in case of extrapolation.
            # Should we insert one uncertainty for each interpolation node outside
            # the original range or should we insert simply each one at the beginning
            # and at the end of the originally provided vector of uncertainties?!
            raise NotImplementedError(
                "Since we are not sure yet about the desired behaviour of returning "
                "sensitivities for extrapolation without assuming constancy, "
                "this feature is not yet implemented. Get in touch with us, "
                "if you need it to discuss how to proceed."
            )

    # Inter- or extrapolate values in the desired fashion relying on SciPy.
    interp_y = interp1d(t, y, fill_value=fill_value, **interp1d_params)
    y_new = interp_y(t_new)

    if kind in ("previous", "next", "nearest"):
        if return_c:
            raise ValueError(
                "Returning the sensitivity matrix is only supported for interpolation "
                "types other than 'previous', 'next' and 'nearest'."
            )
        # Look up uncertainties.
        interp_uy = interp1d(t, uy, fill_value=fill_unc, **interp1d_params)
        uy_new = interp_uy(t_new)
    elif kind == "linear":
        # Calculate boolean arrays of indices from t_new which are outside t's bounds...
        extrap_range_below = t_new < np.min(t)
        extrap_range_above = t_new > np.max(t)
        extrap_range = extrap_range_below | extrap_range_above
        # .. and inside t's bounds.
        interp_range = ~extrap_range

        # Initialize the result array for the standard uncertainties.
        uy_new = np.empty_like(y_new)

        # Initialize the sensitivity matrix of shape (M, N) if needed.
        if return_c:
            C = np.zeros((len(t_new), len(uy)), "float64")

        # First extrapolate the according values if required and then
        # compute interpolated uncertainties following White, 2017.

        # If extrapolation is needed, fill in the values provided via fill_unc.
        if np.any(extrap_range):
            # At this point fill_unc is either a float (np.nan is a float as well) or
            # a 2-tuple of floats. In case we have one float we set uy to this value
            # inside the extrapolation range.
            if isinstance(fill_unc, float):
                uy_new[extrap_range] = fill_unc
            else:
                # Now fill_unc should be a 2-tuple, which we can fill into uy_new.
                uy_new[extrap_range_below], uy_new[extrap_range_above] = fill_unc

            if return_c:
                # In each row of C corresponding to an extrapolation value below the
                # original range set the first column to 1 and in each row of C
                # corresponding to an extrapolation value above the original range set
                # the last column to 1.
                C[:, 0], C[:, -1] = extrap_range_below, extrap_range_above

        # If interpolation is needed, compute uncertainties following White, 2017.
        if np.any(interp_range):
            # This following section is taken partly from scipy.interpolate.interp1d to
            # determine the indices of the relevant original timestamps (or frequencies)
            # for the interpolation.
            # --------------------------------------------------------------------------
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
            # --------------------------------------------------------------------------
            if return_c:
                # Prepare the sensitivity coefficients, which in the first place
                # inside the interpolation range are the Lagrangian polynomials. We
                # compute the Lagrangian polynomials for all interpolation nodes
                # inside the original range.
                L_1 = (t_new[interp_range] - t_hi) / (t_hi - t_lo)
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
                # Compute the uncertainties.
                uy_new[interp_range] = np.sqrt((C @ np.diag(uy ** 2) @ C.T).diagonal())[
                    interp_range
                ]
            else:
                # Since we do not need the sensitivity matrix, we compute
                # uncertainties more efficient.
                uy_prev_sqr = uy[lo] ** 2
                uy_next_sqr = uy[hi] ** 2
                uy_new[interp_range] = np.sqrt(
                    (t_new[interp_range] - t_hi) ** 2 * uy_prev_sqr
                    + (t_new[interp_range] - t_lo) ** 2 * uy_next_sqr
                ) / (t_hi - t_lo)
    else:
        raise NotImplementedError(
            "%s is unsupported yet. Let us know, that you need it." % kind
        )

    if return_c:
        return t_new, y_new, uy_new, C
    return t_new, y_new, uy_new
