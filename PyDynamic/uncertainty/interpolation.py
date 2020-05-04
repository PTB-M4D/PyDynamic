# -*- coding: utf-8 -*-
"""
The :mod:`PyDynamic.uncertainty.interpolation` module implements methods for
the propagation of uncertainties in the application of standard interpolation methods
as provided by :class:`scipy.interpolate.interp1d`.

This module contains the following function:

* :func:`interp1d_unc`: Interpolate arbitrary time series considering the associated
  uncertainties
"""

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

__all__ = ["interp1d_unc"]


def interp1d_unc(
    t_new: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    uy: np.ndarray,
    kind: Optional[str] = "linear",
    bounds_error: Optional[bool] = None,
    fill_value: Optional[bool] = np.nan,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Interpolate a 1-D function considering the associated uncertainties

    t and y are arrays of values used to approximate some function :math:`f \colon y
    = f(x)`.

    Note that calling :func:`interp1d_unc` with NaNs present in input values
    results in undefined behaviour.

    At least two of each of the original timestamps (or frequencies), values and
    associated uncertainties are required and an equal number of each of these three.

    Parameters
    ----------
        t_new : (M,) array_like
            A 1-D array of real values representing the timestamps (or frequencies) at
            which to evaluate the interpolated values.
        t : (N,) array_like
            A 1-D array of real values representing timestamps (or frequencies) in
            ascending order.
        y : (N,) array_like
            A 1-D array of real values. The length of y must be equal to the length
            of t_new
        uy : (N,) array_like
            A 1-D array of real values representing the uncertainties associated with y.
        kind : str, optional
            Specifies the kind of interpolation for y as a string ('previous',
            'next', 'nearest' or 'linear'). Default is ‘linear’.
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
            - If “extrapolate”, then points outside the data range will be extrapolated.

    Returns
    -------
        t_new : (M,) array_like
            interpolation timestamps
        y_new : (M,) array_like
            interpolated measurement values
        uy_new : (M,) array_like
            interpolated measurement values' uncertainties

    References
    ----------
        * White [White2017]_
    """
    # Check for ascending order of timestamps.
    if not np.all(t[1:] >= t[:-1]):
        raise ValueError("Array of timestamps needs to be in ascending order.")
    # Check for proper dimensions of inputs which are not checked as desired by SciPy.
    if not len(y) == len(uy):
        raise ValueError(
            "Array of associated measurement values' uncertainties must be same length "
            "as array of measurement values."
        )
    # Interpolate measurement values in the desired fashion.
    interp_y = interp1d(
        t, y, kind=kind, bounds_error=bounds_error, fill_value=fill_value
    )
    y_new = interp_y(t_new)

    if kind in ("previous", "next", "nearest"):
        # Look up uncertainties in cases where it is applicable.
        interp_uy = interp1d(t, uy, kind=kind)
        uy_new = interp_uy(t_new)
    elif kind == "linear":
        # Determine the relevant interpolation intervals for all interpolation
        # timestamps. We determine the intervals for each timestamp in t_new by
        # finding the biggest of all timestamps in t smaller or equal than the one in
        # t_new. From the array of indices of our left interval bounds we get the
        # right bounds by just incrementing the indices, which of course
        # results in index errors at the end of our array, in case the biggest
        # timestamps in t_new are (quasi) equal to the biggest (i.e. last) timestamp
        # in t. This gets corrected just after the iteration by simply manually
        # choosing one interval "further left", which will just at the node result
        # in the same interpolation (being the actual value of y).
        indices = np.empty_like(t_new, dtype=int)
        it_t_new = np.nditer(t_new, flags=["f_index"])
        while not it_t_new.finished:
            # Find indices of biggest of all timestamps smaller or equal
            # than current interpolation timestamp. Assume that timestamps are in
            # ascending order.
            indices[it_t_new.index] = np.where(t <= it_t_new[0])[0][-1]
            it_t_new.iternext()
        # Correct all degenerated intervals. This happens when the last timestamps of
        # t and t_new are equal.
        indices[np.where(indices == len(t) - 1)] -= 1
        t_prev = t[indices]
        t_next = t[indices + 1]
        # Look up corresponding input uncertainties.
        uy_prev_sqr = uy[indices] ** 2
        uy_next_sqr = uy[indices + 1] ** 2
        # Compute uncertainties for interpolated measurement values.
        uy_new = np.sqrt(
            (t_new - t_next) ** 2 * uy_prev_sqr + (t_new - t_prev) ** 2 * uy_next_sqr
        ) / (t_next - t_prev)
    else:
        raise NotImplementedError

    return t_new, y_new, uy_new
