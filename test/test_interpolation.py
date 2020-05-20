from typing import Dict, Optional, Tuple, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given
from hypothesis.strategies import composite
from numpy.testing import assert_allclose
from pytest import raises

from PyDynamic.uncertainty.interpolation import interp1d_unc


@composite
def timestamps_values_uncertainties_kind(
    draw,
    min_count: Optional[int] = 2,
    max_count: Optional[int] = None,
    kind_tuple: Optional[Tuple[str]] = ("linear", "previous", "next", "nearest"),
    sorted_timestamps: Optional[bool] = True,
    extrapolate: Optional[Union[bool, str]] = False,
    restrict_fill_value: Optional[str] = None,
    restrict_fill_unc: Optional[str] = None,
    returnC: Optional[bool] = False,
) -> Dict[str, Union[np.ndarray, str]]:
    """Set custom strategy for _hypothesis_ to draw desired input from

    Parameters
    ----------
        draw : callable
            this is a hypothesis internal callable to actually draw from provided
            strategies
        min_count : int
            the minimum number of elements expected inside the arrays of timestamps
             (or frequencies), measurement values and associated uncertainties
        max_count : int
            the maximum number of elements expected inside the arrays of timestamps
            (or frequencies), measurement values and associated uncertainties
        kind_tuple : tuple(str), optional
            the tuple of strings out of "linear", "previous", "next", "nearest",
            "spline", "least-squares" from which the strategy for the
            kind randomly chooses. Defaults to the valid options "linear",
            "previous", "next", "nearest"
        sorted_timestamps : bool
            if True the timestamps (or frequencies) are guaranteed to be in ascending
            order, if False they still might be by coincidence or not
        extrapolate : bool or str, optional
            If True the interpolation timestamps (or frequencies) are generated such
            that extrapolation is necessary by guarantying at least one of the
            interpolation timestamps (or frequencies) outside the original bounds
            and accordingly setting appropriate values for `fill_value` and
            `bounds_error = False`. If False each element of t_new is guaranteed to
            lie within the range of t. Can be set to "above" or "below" to guarantee
            at least one element of t_new to lie either below or above the bounds of t.
        restrict_fill_value : str, optional
            String specifying the desired strategy for drawing a fill_value. One of
            "float", "tuple", "str", "nan" to guarantee either a float, a tuple of
            two floats, the string "extrapolate" or np.nan.
        restrict_fill_unc : str, optional
            Same as fill_value, but just for the uncertainties.
        returnC : bool, optional
            If True we request the sensitivities to be returned. If False we do not
            request them. Defaults to False.

    Returns
    -------
        A dict containing the randomly generated expected input parameters for
        `interp1d_unc()`
    """

    def draw_fill_values(strategy_spec: str):
        """Little helper to find proper strategy for efficient testing.

        Parameters
        ----------
            strategy_spec : str
                String specifying the desired strategy for drawing a fill_value. One of
                "float", "tuple", "str", "nan" to guarantee either a float, a tuple of
                two floats, the string "extrapolate" or np.nan.

        Returns
        -------
            The drawn sample to match desired fill_value.
        """
        float_strategy = st.floats(**float_generic_params)
        tuple_strategy = st.tuples(float_strategy, float_strategy)
        string_strategy = st.just("extrapolate")
        nan_strategy = st.just(np.nan)
        if strategy_spec == "float":
            fill_strategy = float_strategy
        elif strategy_spec == "tuple":
            fill_strategy = tuple_strategy
        elif strategy_spec == "str":
            fill_strategy = string_strategy
        elif strategy_spec == "nan":
            fill_strategy = nan_strategy
        else:
            fill_strategy = st.one_of(
                float_strategy, tuple_strategy, string_strategy, nan_strategy
            )
        return draw(fill_strategy)

    # Set the maximum absolute value for floats to be really unique in calculations.
    float_abs_max = 1e64
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
        fill_value = fill_unc = np.nan
        bounds_error = True
    else:
        # In case we want to extrapolate, draw some fill values for the
        # out-of-bounds range. Those will be either single floats or a 2-tuple of
        # floats or the special value "extrapolate".
        fill_value = draw_fill_values(restrict_fill_value)
        fill_unc = draw_fill_values(restrict_fill_unc)
        bounds_error = False

    # Draw interpolation timestamps (or frequencies).
    t_new = draw(hnp.arrays(**strategy_params))

    if extrapolate:
        # In case we want to extrapolate, make sure we actually do after having drawn
        # the timestamps (or frequencies) not to randomly have drawn values inside
        # original bounds and if even more constraints are given ensure those.
        assume(np.min(t_new) < np.min(t) or np.max(t_new) > np.max(t))
        if extrapolate == "above":
            assume(np.max(t_new) > np.max(t))
        else:
            assume(np.min(t_new) < np.min(t))

    kind = draw(st.sampled_from(kind_tuple))
    assume_sorted = sorted_timestamps
    return {
        "t_new": t_new,
        "t": t,
        "y": y,
        "uy": uy,
        "kind": kind,
        "fill_value": fill_value,
        "fill_unc": fill_unc,
        "bounds_error": bounds_error,
        "assume_sorted": assume_sorted,
        "returnC": returnC,
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
    # Check that extrapolation is executable in general.
    assert interp1d_unc(**interp_inputs)


@given(
    timestamps_values_uncertainties_kind(
        sorted_timestamps=True, extrapolate="below", restrict_fill_value="str"
    )
)
def test_extrapolate_below_without_fill_value_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is below the minimum of t and
    # fill_value=="extrapolate", which means constant extrapolation from the boundaries.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works, meaning in the present case, that the boundary
    # value of y is taken for all t_new below the original bound.
    assert np.all(
        y_new[interp_inputs["t_new"] < np.min(interp_inputs["t"])]
        == interp_inputs["y"][0]
    )


@given(
    timestamps_values_uncertainties_kind(
        extrapolate="below", restrict_fill_value="float"
    )
)
def test_extrapolate_below_with_fill_value_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is below the minimum of t and
    # fill_value is a float, which means constant extrapolation with this value.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works.
    assert np.all(
        y_new[interp_inputs["t_new"] < np.min(interp_inputs["t"])]
        == interp_inputs["fill_value"]
    )


@given(
    timestamps_values_uncertainties_kind(
        extrapolate="below", restrict_fill_value="tuple"
    )
)
def test_extrapolate_below_with_fill_values_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is below the minimum of t and
    # fill_value is a tuple, which means constant extrapolation with its first
    # element.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works.
    assert np.all(
        y_new[interp_inputs["t_new"] < np.min(interp_inputs["t"])]
        == interp_inputs["fill_value"][0]
    )


@given(
    timestamps_values_uncertainties_kind(
        sorted_timestamps=True, extrapolate="above", restrict_fill_value="str"
    )
)
def test_extrapolate_above_without_fill_value_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is above the maximum of t and
    # fill_value=="extrapolate", which means constant extrapolation from the boundaries.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works, meaning in the present case, that the boundary
    # value of y is taken for all t_new above the original bound.
    assert np.all(
        y_new[interp_inputs["t_new"] > np.max(interp_inputs["t"])]
        == interp_inputs["y"][-1]
    )


@given(
    timestamps_values_uncertainties_kind(
        extrapolate="above", restrict_fill_value="float"
    )
)
def test_extrapolate_above_with_fill_value_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is above the maximum of t and
    # fill_value is a float, which means constant extrapolation with this value.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works.
    assert np.all(
        y_new[interp_inputs["t_new"] > np.max(interp_inputs["t"])]
        == interp_inputs["fill_value"]
    )


@given(
    timestamps_values_uncertainties_kind(
        extrapolate="above", restrict_fill_value="tuple"
    )
)
def test_extrapolate_above_with_fill_values_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is above the maximum of t and
    # fill_value is a tuple, which means constant extrapolation with its second element.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works.
    assert np.all(
        y_new[interp_inputs["t_new"] > np.max(interp_inputs["t"])]
        == interp_inputs["fill_value"][1]
    )


@given(
    timestamps_values_uncertainties_kind(
        sorted_timestamps=True, extrapolate="below", restrict_fill_unc="str"
    )
)
def test_extrapolate_below_without_fill_unc_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is below the minimum of t and
    # fill_unc=="extrapolate", which means constant extrapolation from the boundaries.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works, meaning in the present case, that the boundary
    # value of y is taken for all t_new below the original bound.
    assert np.all(
        uy_new[interp_inputs["t_new"] < np.min(interp_inputs["t"])]
        == interp_inputs["uy"][0]
    )


@given(
    timestamps_values_uncertainties_kind(extrapolate="below", restrict_fill_unc="float")
)
def test_extrapolate_below_with_fill_unc_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is below the minimum of t and
    # fill_unc is a float, which means constant extrapolation with this value.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works.
    assert np.all(
        uy_new[interp_inputs["t_new"] < np.min(interp_inputs["t"])]
        == interp_inputs["fill_unc"]
    )


@given(
    timestamps_values_uncertainties_kind(extrapolate="below", restrict_fill_unc="tuple")
)
def test_extrapolate_below_with_fill_uncs_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is below the minimum of t and
    # fill_unc is a tuple, which means constant extrapolation with its first element.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works.
    assert np.all(
        uy_new[interp_inputs["t_new"] < np.min(interp_inputs["t"])]
        == interp_inputs["fill_unc"][0]
    )


@given(
    timestamps_values_uncertainties_kind(
        sorted_timestamps=True, extrapolate="above", restrict_fill_unc="str"
    )
)
def test_extrapolate_above_without_fill_unc_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is above the maximum of t and
    # fill_unc=="extrapolate", which means constant extrapolation from the boundaries.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works, meaning in the present case, that the boundary
    # value of y is taken for all t_new above the original bound.
    assert np.all(
        uy_new[interp_inputs["t_new"] > np.max(interp_inputs["t"])]
        == interp_inputs["uy"][-1]
    )


@given(
    timestamps_values_uncertainties_kind(extrapolate="above", restrict_fill_unc="float")
)
def test_extrapolate_above_with_fill_unc_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is above the maximum of t and
    # fill_unc is a float, which means constant extrapolation with this value.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works.
    assert np.all(
        uy_new[interp_inputs["t_new"] > np.max(interp_inputs["t"])]
        == interp_inputs["fill_unc"]
    )


@given(
    timestamps_values_uncertainties_kind(extrapolate="above", restrict_fill_unc="tuple")
)
def test_extrapolate_above_with_fill_uncs_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of t_new is above the maximum of t and
    # fill_unc is a tuple, which means constant extrapolation with its second element.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works.
    assert np.all(
        uy_new[interp_inputs["t_new"] > np.max(interp_inputs["t"])]
        == interp_inputs["fill_unc"][1]
    )


@given(timestamps_values_uncertainties_kind(returnC=True, kind_tuple=("linear",)))
def test_compare_returnc_interp1d_unc(interp_inputs):
    # Compare the uncertainties computed from the sensitivities inside the
    # interpolation range and directly.
    uy_new_with_sensitivities = interp1d_unc(**interp_inputs)[2]
    interp_inputs["returnC"] = False
    uy_new_without_sensitivities = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation results match up to machine epsilon.
    assert_allclose(uy_new_with_sensitivities, uy_new_without_sensitivities, rtol=9e-15)


@given(
    timestamps_values_uncertainties_kind(
        returnC=True, extrapolate=True, kind_tuple=("linear",)
    )
)
def test_failing_returnc_with_extrapolation_interp1d_unc(interp_inputs):
    # Since we have not implemented these cases, for now we
    # check for exception being thrown.
    assume(not isinstance(interp_inputs["fill_unc"], str))
    with raises(NotImplementedError):
        interp1d_unc(**interp_inputs)


@given(
    timestamps_values_uncertainties_kind(
        returnC=True, extrapolate=True, kind_tuple=("linear",), restrict_fill_unc="str"
    )
)
def test_returnc_with_extrapolation_interp1d_unc(interp_inputs):
    # Check if extrapolation with constant values outside interpolation range and
    # returning of sensitivities is callable.
    assert interp1d_unc(**interp_inputs)


@given(
    timestamps_values_uncertainties_kind(
        returnC=True,
        extrapolate=True,
        kind_tuple=("linear",),
        restrict_fill_unc="str",
        sorted_timestamps=True,
    )
)
def test_returnc_with_extrapolation_check_below_bound_interp1d_unc(interp_inputs):
    # Check if extrapolation with constant values outside interpolation range and
    # returning sensitivities work as expected regarding extrapolation values
    # below original bound.
    uy_new, C = interp1d_unc(**interp_inputs)[2:]
    assert np.all(
        uy_new[interp_inputs["t_new"] < np.min(interp_inputs["t"])]
        == interp_inputs["uy"][0]
    )


@given(
    timestamps_values_uncertainties_kind(
        returnC=True,
        extrapolate=True,
        kind_tuple=("linear",),
        restrict_fill_unc="str",
        sorted_timestamps=True,
    )
)
def test_returnc_with_extrapolation_check_uy_new_above_bound_interp1d_unc(
    interp_inputs,
):
    # Check if extrapolation with constant values outside interpolation range and
    # returning sensitivities work as expected regarding extrapolation values
    # above original bound.
    uy_new = interp1d_unc(**interp_inputs)[2]
    assert np.all(
        uy_new[interp_inputs["t_new"] > np.max(interp_inputs["t"])]
        == interp_inputs["uy"][-1]
    )


@given(
    timestamps_values_uncertainties_kind(
        returnC=True, extrapolate=True, kind_tuple=("linear",), restrict_fill_unc="str",
    )
)
def test_returnc_with_extrapolation_check_c_interp1d_unc(interp_inputs,):
    # Check if sensitivity computation parallel to linear interpolation and
    # extrapolation with constant values works as expected regarding the shape and
    # content of the sensitivity matrix.
    C = interp1d_unc(**interp_inputs)[3]

    # Check that C has the right shape.
    assert C.shape == (len(interp_inputs["t_new"]), len(interp_inputs["t"]))

    # Find interpolation range because we reuse it.
    interp_range = (interp_inputs["t_new"] >= np.min(interp_inputs["t"])) | (
        interp_inputs["t_new"] <= np.max(interp_inputs["t"])
    )

    # Check if each row corresponding to an extrapolated value contains exactly one
    # non-zero sensitivity.
    assert np.all(np.count_nonzero(C[np.where(~interp_range)], 1) == 1)

    # Check if each row corresponding to an interpolated value contains either exactly
    # one or exactly two non-zero sensitivities, which are the two possible cases
    # when performing Lagrangian linear interpolation.
    assert np.all(
        np.any(
            (
                np.count_nonzero(C[np.where(interp_range)], 1) == 2,
                np.count_nonzero(C[np.where(interp_range)], 1) == 1,
            ),
            0,
        )
    )

    # Check if each row of sensitivities sum to one, which should hold for the
    # Lagrangians and proves equality with one for extrapolation sensitivities.
    assert_allclose(np.sum(C, 1), np.ones_like(interp_inputs["t_new"]))


@given(
    timestamps_values_uncertainties_kind(
        returnC=True, kind_tuple=("previous", "next", "nearest",)
    )
)
def test_value_error_for_returnc_interp1d_unc(interp_inputs):
    # Check erroneous calls with returnC and wrong kind.
    with raises(NotImplementedError):
        interp1d_unc(**interp_inputs)


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
        interp_inputs["uy"] = np.tile(interp_inputs["uy"], 2)
        interp1d_unc(**interp_inputs)


@given(timestamps_values_uncertainties_kind(kind_tuple=("spline", "least-squares")))
def test_raise_not_implemented_yet_interp1d(interp_inputs):
    # Check that not implemented versions raise exceptions.
    with raises(NotImplementedError):
        interp1d_unc(**interp_inputs)


@given(timestamps_values_uncertainties_kind(extrapolate=True))
def test_raise_value_error_interp1d_unc(interp_inputs):
    # Check that interpolation with points outside the original domain raises
    # exception if requested.
    interp_inputs["bounds_error"] = True
    with raises(ValueError):
        interp1d_unc(**interp_inputs)
