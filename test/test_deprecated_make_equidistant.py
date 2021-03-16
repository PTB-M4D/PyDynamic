# -*- coding: utf-8 -*-
""" Perform tests on the method *make_equidistant*."""
from typing import Dict, Optional, Tuple, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis.strategies import composite
from pytest import deprecated_call, raises

from PyDynamic.misc.tools import make_equidistant


@composite
def timestamps_values_uncertainties_kind(
    draw,
    min_count: Optional[int] = 2,
    max_count: Optional[int] = None,
    kind_tuple: Optional[Tuple[str]] = ("linear", "previous", "next", "nearest"),
    sorted_timestamps: Optional[bool] = True,
) -> Dict[str, Union[np.ndarray, str]]:
    """Set custom strategy for _hypothesis_ to draw desired input from

    Parameters
    ----------
        draw: callable
            this is a hypothesis internal callable to actually draw from provided
            strategies
        min_count: int
            the minimum number of elements expected inside the arrays of timestamps,
            measurement values and associated uncertainties
        max_count: int
            the maximum number of elements expected inside the arrays of timestamps,
            measurement values and associated uncertainties
        kind_tuple: tuple(str), optional
            the tuple of strings out of "linear", "previous", "next", "nearest",
            "spline", "least-squares" from which the strategy for the
            kind randomly chooses. Defaults to the valid options "linear",
            "previous", "next", "nearest".
        sorted_timestamps: bool
            if the timestamps should be in ascending order or not

    Returns
    -------
        A dict containing the randomly generated expected input parameters t, y, uy,
        dt, kind for make_equidistant()
    """
    # Set the maximum absolute value for floats to be really unique in calculations.
    float_abs_max = 1e64
    # Set all common parameters for timestamps, measurements values and associated
    # uncertainties including allowed ranges and number of elements.
    shape_for_timestamps = hnp.array_shapes(
        max_dims=1, min_side=min_count, max_side=max_count
    )
    strategy_params = {
        "dtype": np.float,
        "shape": shape_for_timestamps,
        "elements": st.floats(
            min_value=-float_abs_max,
            max_value=float_abs_max,
            allow_nan=False,
            allow_infinity=False,
        ),
        "unique": True,
    }
    # Draw "original" timestamps.
    t = draw(hnp.arrays(**strategy_params))
    # Sort timestamps in ascending order.
    if sorted_timestamps:
        t.sort()
    # Reuse "original" timestamps shape for measurements values and associated
    # uncertainties and draw both.
    strategy_params["shape"] = np.shape(t)
    y = draw(hnp.arrays(**strategy_params))
    uy = draw(hnp.arrays(**strategy_params))
    dt = draw(
        st.floats(
            min_value=(np.max(t) - np.min(t)) * 1e-3,
            max_value=(np.max(t) - np.min(t)) / 2,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    kind = draw(st.sampled_from(kind_tuple))
    return {"t": t, "y": y, "uy": uy, "dt": dt, "kind": kind}


@given(timestamps_values_uncertainties_kind())
def test_deprecated_call_make_equidistant(interp_inputs):
    with deprecated_call():
        make_equidistant(**interp_inputs)


# noinspection PyArgumentList
@given(timestamps_values_uncertainties_kind())
def test_too_short_call_make_equidistant(interp_inputs):
    # Check erroneous calls with too few inputs.
    with raises(TypeError):
        make_equidistant(interp_inputs["t"])
        make_equidistant(interp_inputs["t"], interp_inputs["y"])


@given(timestamps_values_uncertainties_kind())
def test_full_call_make_equidistant(interp_inputs):
    t_new, y_new, uy_new = make_equidistant(**interp_inputs)
    # Check the equal dimensions of the minimum calls output.
    assert len(t_new) == len(y_new) == len(uy_new)


@given(timestamps_values_uncertainties_kind())
def test_wrong_input_lengths_call_make_equidistant(interp_inputs):
    # Check erroneous calls with unequally long inputs.
    with raises(ValueError):
        y_wrong = np.tile(interp_inputs["y"], 2)
        uy_wrong = np.tile(interp_inputs["uy"], 3)
        make_equidistant(interp_inputs["t"], y_wrong, uy_wrong)


@given(timestamps_values_uncertainties_kind())
def test_t_new_to_dt_make_equidistant(interp_inputs):
    t_new = make_equidistant(**interp_inputs)[0]
    delta_t_new = np.diff(t_new)
    # Check if the new timestamps are ascending.
    assert not np.any(delta_t_new < 0)


@given(timestamps_values_uncertainties_kind(kind_tuple=("previous", "next", "nearest")))
def test_prev_in_make_equidistant(interp_inputs):
    y_new, uy_new = make_equidistant(**interp_inputs)[1:3]
    # Check if all 'interpolated' values are present in the actual values.
    assert np.all(np.isin(y_new, interp_inputs["y"]))
    assert np.all(np.isin(uy_new, interp_inputs["uy"]))


@given(timestamps_values_uncertainties_kind(kind_tuple=["linear"]))
def test_linear_in_make_equidistant(interp_inputs):
    y_new, uy_new = make_equidistant(**interp_inputs)[1:3]
    # Check if all interpolated values lie in the range of the original values.
    assert np.all(np.amin(interp_inputs["y"]) <= y_new)
    assert np.all(np.amax(interp_inputs["y"]) >= y_new)


@given(st.integers(min_value=3, max_value=1000))
def test_linear_uy_in_make_equidistant(n):
    # Check for given input, if interpolated uncertainties equal 1 and
    # :math:`sqrt(2) / 2`.
    dt_unit = 2
    t_unit = np.arange(0, n, dt_unit)
    y = uy_unit = np.ones_like(t_unit)
    dt_half = dt_unit / 2
    uy_new = make_equidistant(t_unit, y, uy_unit, dt_half, "linear")[2]
    assert np.all(uy_new[0:n:dt_unit] == 1) and np.all(
        uy_new[1:n:dt_unit] == np.sqrt(2) / 2
    )


@given(timestamps_values_uncertainties_kind(kind_tuple=("spline", "least-squares")))
def test_raise_not_implemented_yet_make_equidistant(interp_inputs):
    # Check that not implemented versions raise exceptions.
    with raises(NotImplementedError):
        make_equidistant(**interp_inputs)
