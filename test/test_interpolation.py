import itertools
from typing import Dict, Optional, Tuple, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given
from hypothesis.strategies import composite
from pytest import raises

from PyDynamic.uncertainty.interpolation import interp1d_unc


@composite
def timestamps_values_uncertainties_kind(
    draw,
    min_count: Optional[int] = 2,
    max_count: Optional[int] = None,
    kind_tuple: Optional[Tuple[str]] = ("linear", "previous", "next", "nearest"),
    sorted_timestamps: Optional[bool] = True,
    extrapolate: Optional[bool] = False,
) -> Dict[str, Union[np.ndarray, str]]:
    """Set custom strategy for _hypothesis_ to draw desired input from

    Parameters
    ----------
        draw: callable
            this is a hypothesis internal callable to actually draw from provided
            strategies
        min_count: int
            the minimum number of elements expected inside the arrays of timestamps
             (or frequencies), measurement values and associated uncertainties
        max_count: int
            the maximum number of elements expected inside the arrays of timestamps
            (or frequencies), measurement values and associated uncertainties
        kind_tuple: tuple(str), optional
            the tuple of strings out of "linear", "previous", "next", "nearest",
            "spline", "lagrange", "least-squares" from which the strategy for the
            kind randomly chooses. Defaults to the valid options "linear",
            "previous", "next", "nearest"
        sorted_timestamps: bool
            if True the timestamps (or frequencies) are guaranteed to be in ascending
            order, if False they still might be by coincidence or not
        extrapolate: bool
            if True the interpolation timestamps (or frequencies) are generated such
            that extrapolation is necessary by guarantying at least one of the
            interpolation timestamps (or frequencies) outside the original bounds
            and accordingly setting appropriate values for `fill_value` and
            `bounds_error = False`

    Returns
    -------
        A dict containing the randomly generated expected input parameters t, y, uy,
        dt, kind for interp1d_unc()
    """
    # Set the maximum absolute value for floats to be really unique in calculations.
    float_abs_max = 1e300
    # Set generic float parameters.
    float_generic_params = {
        "allow_nan": False,
        "allow_infinity": False,
    }
    # Set all common parameters for timestamps (or frequencies), measurements values
    # and associated uncertainties.
    shape_for_timestamps = hnp.array_shapes(
        max_dims=1, min_side=min_count, max_side=max_count
    )
    strategy_params = {
        "dtype": np.float,
        "shape": shape_for_timestamps,
        "elements": st.floats(
            min_value=-float_abs_max, max_value=float_abs_max, **float_generic_params
        ),
        "unique": True,
    }
    # Draw "original" timestamps (or frequencies).
    t = draw(hnp.arrays(**strategy_params))
    # Sort timestamps (or frequencies) in ascending order.
    if sorted_timestamps:
        ind = np.argsort(t)
        t = t[ind]

    # Reuse "original" timestamps (or frequencies) shape for measurements values and
    # associated uncertainties and draw both.
    strategy_params["shape"] = np.shape(t)
    y = draw(hnp.arrays(**strategy_params))
    uy = draw(hnp.arrays(**strategy_params))

    # Reset shape for interpolation timestamps (or frequencies).
    strategy_params["shape"] = shape_for_timestamps
    # Look up minimum and maximum of original timestamps (or frequencies) just once.
    t_min = np.min(t)
    t_max = np.max(t)

    if not extrapolate:
        # In case we do not want to extrapolate, use range of "original" timestamps (or
        # frequencies) as boundaries.
        strategy_params["elements"] = st.floats(
            min_value=t_min, max_value=t_max, **float_generic_params
        )
        fill_value = None
        bounds_error = True
    else:
        # In case we want to extrapolate, draw some fill values for the
        # out-of-bounds range. Those will be either single floats or a 2-tuple of
        # floats or the special value "extrapolate".
        float_strategy = st.floats(**float_generic_params)
        fill_value = draw(
            st.one_of(
                float_strategy,
                st.tuples(
                    *tuple(itertools.repeat(st.floats(**float_generic_params), 2))
                ),
                st.just("extrapolate"),
            )
        )
        bounds_error = False

    # Draw interpolation timestamps (or frequencies).
    t_new = draw(hnp.arrays(**strategy_params))

    if extrapolate:
        # In case we want to extrapolate, make sure we actually do after having drawn
        # the timestamps (or frequencies) not to randomly have drawn values inside
        # original bounds.
        assume(np.min(t_new) < np.min(t) or np.max(t_new) > np.max(t))

    kind = draw(st.sampled_from(kind_tuple))
    assume_sorted = sorted_timestamps
    return {
        "t_new": t_new,
        "t": t,
        "y": y,
        "uy": uy,
        "kind": kind,
        "fill_value": fill_value,
        "bounds_error": bounds_error,
        "assume_sorted": assume_sorted,
    }


@given(timestamps_values_uncertainties_kind())
def test_usual_call(interp_inputs):
    t_new, y_new, uy_new = interp1d_unc(**interp_inputs)
    # Check the equal dimensions of the minimum calls output.
    assert len(t_new) == len(y_new) == len(uy_new)


@given(timestamps_values_uncertainties_kind())
def test_wrong_input_length_y_call_interp1d_unc(interp_inputs):
    # Check erroneous calls with unequally long inputs.
    interp_inputs["y"] = np.tile(interp_inputs["y"], 2)
    with raises(ValueError):
        interp1d_unc(**interp_inputs)


@given(timestamps_values_uncertainties_kind())
def test_wrong_input_length_uy_call_interp1d_unc(interp_inputs):
    # Check erroneous calls with unequally long inputs.
    interp_inputs["uy"] = np.tile(interp_inputs["uy"], 2)
    with raises(ValueError):
        interp1d_unc(**interp_inputs)


@given(timestamps_values_uncertainties_kind(sorted_timestamps=False))
def test_wrong_input_order_call_interp1d_unc(interp_inputs):
    # Ensure the timestamps are not in ascending order.
    assume(not np.all(interp_inputs["t"][1:] >= interp_inputs["t"][:-1]))
    # Check erroneous calls with descending timestamps.
    interp_inputs["assume_sorted"] = True
    with raises(ValueError):
        # Reverse order of t and call interp1d_unc().
        interp1d_unc(**interp_inputs)


@given(timestamps_values_uncertainties_kind(kind_tuple=("previous", "next", "nearest")))
def test_trivial_in_interp1d_unc(interp_inputs):
    y_new, uy_new = interp1d_unc(**interp_inputs)[1:3]
    # Check if all 'interpolated' values are present in the actual values.
    assert np.all(np.isin(y_new, interp_inputs["y"]))
    assert np.all(np.isin(uy_new, interp_inputs["uy"]))


@given(timestamps_values_uncertainties_kind(kind_tuple=["linear"]))
def test_linear_in_interp1d_unc(interp_inputs):
    y_new, uy_new = interp1d_unc(**interp_inputs)[1:3]
    # Check if all interpolated values lie in the range of the original values.
    assert np.all(np.min(interp_inputs["y"]) <= y_new)
    assert np.all(np.max(interp_inputs["y"]) >= y_new)


@given(timestamps_values_uncertainties_kind(extrapolate=True))
def test_extrapolate_interp1d_unc(interp_inputs):
    # Check that extrapolation works.
    assert interp1d_unc(**interp_inputs)


@given(st.integers(min_value=3, max_value=1000))
def test_linear_uy_in_interp1d_unc(n,):
    # Check for given input, if interpolated uncertainties equal 1 and
    # :math:`sqrt(2) / 2`.
    dt_unit = 2
    dt_half = dt_unit / 2
    t_new = np.arange(0, n, dt_half)
    t_unit = np.arange(0, n + dt_half, dt_unit)
    y = uy_unit = np.ones_like(t_unit)
    uy_new = interp1d_unc(t_new, t_unit, y, uy_unit, "linear")[2]
    assert np.all(uy_new[0:n:dt_unit] == 1) and np.all(
        uy_new[1:n:dt_unit] == np.sqrt(2) / 2
    )


@given(timestamps_values_uncertainties_kind())
def test_wrong_input_lengths_call_interp1d(interp_inputs):
    # Check erroneous calls with unequally long inputs.
    with raises(ValueError):
        y_wrong = np.tile(interp_inputs["y"], 2)
        uy_wrong = np.tile(interp_inputs["uy"], 3)
        interp1d_unc(interp_inputs["t"], y_wrong, uy_wrong)


@given(
    timestamps_values_uncertainties_kind(
        kind_tuple=("spline", "lagrange", "least-squares")
    )
)
def test_raise_not_implemented_yet_interp1d(interp_inputs):
    # Check that not implemented versions raise exceptions.
    with raises(NotImplementedError):
        interp1d_unc(**interp_inputs)


@given(timestamps_values_uncertainties_kind(extrapolate=True))
def test_raise_value_error_interp1d_unc(interp_inputs):
    # Check that interpolation with points outside the original domain raises
    # exception.
    interp_inputs["bounds_error"] = True
    with raises(ValueError):
        interp1d_unc(**interp_inputs)
