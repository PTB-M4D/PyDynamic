from typing import Dict, Optional, Tuple, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.strategies import composite
from numpy.testing import assert_allclose
from pytest import raises

from PyDynamic.uncertainty.interpolate import interp1d_unc, make_equidistant


@composite
def values_uncertainties_kind(
    draw,
    min_count: Optional[int] = 4,
    max_count: Optional[int] = None,
    kind_tuple: Optional[Tuple[str]] = (
        "linear",
        "previous",
        "next",
        "nearest",
        "cubic",
    ),
    sorted_xs: Optional[bool] = True,
    extrapolate: Optional[Union[bool, str]] = False,
    restrict_fill_value: Optional[str] = None,
    restrict_fill_unc: Optional[str] = None,
    returnC: Optional[bool] = False,
    for_make_equidistant: Optional[bool] = False,
) -> Dict[str, Union[np.ndarray, str]]:
    """Set custom strategy for _hypothesis_ to draw desired input from

    Parameters
    ----------
        draw : callable
            this is a hypothesis internal callable to actually draw from provided
            strategies
        min_count : int, optional
            the minimum number of elements expected inside the arrays of x and y
            values and associated uncertainties. (default = 2)
        max_count : int, optional
            the maximum number of elements expected inside the arrays of x and y values
            and associated uncertainties (default is None)
        kind_tuple : tuple(str), optional
            the tuple of strings out of "linear", "previous", "next", "nearest",
            "spline", "least-squares" from which the strategy for the
            kind randomly chooses. Defaults to the valid options "linear",
            "previous", "next", "nearest", "cubic"
        sorted_xs : bool, optional
            if True (default) the x values are guaranteed to be in
            ascending order, if False they still might be by coincidence or not
        extrapolate : bool or str, optional
            If True the array to evaluate the interpolant at are generated such that
            extrapolation is necessary by guarantying at least one of the values
            outside the original bounds and accordingly setting appropriate values
            for `fill_value` and `bounds_error = False`. If False (default) each
            element of x_new is guaranteed to lie within the range of x. Can be set
            to "above" or "below" to guarantee at least one element of x_new to lie
            either below or above the bounds of x.
        restrict_fill_value : str, optional
            String specifying the desired strategy for drawing a fill_value. One of
            "float", "tuple", "str", "nan" to guarantee either a float, a tuple of
            two floats, the string "extrapolate" or np.nan. (default is None)
        restrict_fill_unc : str, optional
            Same as fill_value, but just for the uncertainties. (default is None)
        returnC : bool, optional
            If True we request the sensitivities to be returned. If False (default) we
            do not request them.
        for_make_equidistant : bool, optional
            If True we return the expected parameters for calling `make_equidistant()`.
            If False (default) we return the expected parameters for calling
            `interp1d_unc()`.

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
        float_strategy = hst.floats(**float_generic_params)
        tuple_strategy = hst.tuples(float_strategy, float_strategy)
        string_strategy = hst.just("extrapolate")
        nan_strategy = hst.just(np.nan)
        if strategy_spec == "float":
            fill_strategy = float_strategy
        elif strategy_spec == "tuple":
            fill_strategy = tuple_strategy
        elif strategy_spec == "str":
            fill_strategy = string_strategy
        elif strategy_spec == "nan":
            fill_strategy = nan_strategy
        else:
            fill_strategy = hst.one_of(
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
    # Set all common parameters for x and y values and associated uncertainties.
    shape_for_x = hnp.array_shapes(max_dims=1, min_side=min_count, max_side=max_count)
    strategy_params = {
        "dtype": np.float,
        "shape": shape_for_x,
        "elements": hst.floats(
            min_value=-float_abs_max, max_value=float_abs_max, **float_generic_params
        ),
        "unique": True,
    }
    # Draw "original" x values.
    x = draw(hnp.arrays(**strategy_params))
    # Sort x values in ascending order.
    if sorted_xs:
        ind = np.argsort(x)
        x = x[ind]

    # Reuse "original" x values' shape for y values and associated uncertainties and
    # draw both.
    strategy_params["shape"] = np.shape(x)
    y = draw(hnp.arrays(**strategy_params))
    uy = draw(hnp.arrays(**strategy_params))

    # Draw the interpolation kind from the provided tuple.
    kind = draw(hst.sampled_from(kind_tuple))

    if for_make_equidistant:
        dx = draw(
            hst.floats(
                min_value=(np.max(x) - np.min(x)) * 1e-3,
                max_value=(np.max(x) - np.min(x)) / 2,
                exclude_min=True,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return {"x": x, "y": y, "uy": uy, "dx": dx, "kind": kind}
    else:
        # Reset shape for values to evaluate the interpolant at.
        strategy_params["shape"] = shape_for_x
        # Look up minimum and maximum of original x values just once.
        x_min = np.min(x)
        x_max = np.max(x)

        if not extrapolate:
            # In case we do not want to extrapolate, use range of "original"
            # x values as boundaries.
            strategy_params["elements"] = hst.floats(
                min_value=x_min, max_value=x_max, **float_generic_params
            )
            fill_value = fill_unc = np.nan
            # Switch between default value None and intentionally setting to True,
            # which should behave identically.
            bounds_error = draw(hst.one_of(hst.just(True), hst.none()))
        else:
            # In case we want to extrapolate, draw some fill values for the
            # out-of-bounds range. Those will be either single floats or a 2-tuple of
            # floats or the special value "extrapolate".
            fill_value = draw_fill_values(restrict_fill_value)
            fill_unc = draw_fill_values(restrict_fill_unc)
            bounds_error = False

        # Draw values to evaluate the interpolant at.
        x_new = draw(hnp.arrays(**strategy_params))

        if extrapolate:
            # In case we want to extrapolate, make sure we actually do after having
            # drawn the values to evaluate the interpolant at not to randomly have
            # drawn values inside original bounds and if even more constraints are
            # given ensure those.
            assume(np.min(x_new) < np.min(x) or np.max(x_new) > np.max(x))
            if extrapolate == "above":
                assume(np.max(x_new) > np.max(x))
            else:
                assume(np.min(x_new) < np.min(x))

        assume_sorted = sorted_xs
        return {
            "x_new": x_new,
            "x": x,
            "y": y,
            "uy": uy,
            "kind": kind,
            "fill_value": fill_value,
            "fill_unc": fill_unc,
            "bounds_error": bounds_error,
            "assume_sorted": assume_sorted,
            "returnC": returnC,
        }


@given(values_uncertainties_kind())
def test_usual_call_interp1d_unc(interp_inputs):
    t_new, y_new, uy_new = interp1d_unc(**interp_inputs)[:]
    # Check the equal dimensions of the minimum calls output.
    assert len(t_new) == len(y_new) == len(uy_new)


@given(values_uncertainties_kind())
@pytest.mark.slow
def test_wrong_input_length_y_call_interp1d_unc(interp_inputs):
    # Check erroneous calls with unequally long inputs.
    interp_inputs["y"] = np.tile(interp_inputs["y"], 2)
    with raises(ValueError):
        interp1d_unc(**interp_inputs)


@given(values_uncertainties_kind())
@pytest.mark.slow
def test_wrong_input_length_uy_call_interp1d_unc(interp_inputs):
    # Check erroneous calls with unequally long inputs.
    interp_inputs["uy"] = np.tile(interp_inputs["uy"], 2)
    with raises(ValueError):
        interp1d_unc(**interp_inputs)


@given(values_uncertainties_kind(kind_tuple=("previous", "next", "nearest")))
@pytest.mark.slow
def test_trivial_in_interp1d_unc(interp_inputs):
    y_new, uy_new = interp1d_unc(**interp_inputs)[1:3]
    # Check if all 'interpolated' values are present in the actual values.
    assert np.all(np.isin(y_new, interp_inputs["y"]))
    assert np.all(np.isin(uy_new, interp_inputs["uy"]))


@given(values_uncertainties_kind(kind_tuple=["linear"]))
@pytest.mark.slow
def test_linear_in_interp1d_unc(interp_inputs):
    y_new, uy_new = interp1d_unc(**interp_inputs)[1:3]
    # Check if all interpolated values lie in the range of the original values.
    assert np.all(np.min(interp_inputs["y"]) <= y_new)
    assert np.all(np.max(interp_inputs["y"]) >= y_new)


@given(values_uncertainties_kind(extrapolate=True))
@pytest.mark.slow
def test_extrapolate_interp1d_unc(interp_inputs):
    # Check that extrapolation is executable in general.
    assert interp1d_unc(**interp_inputs)


@given(
    values_uncertainties_kind(
        sorted_xs=True, extrapolate="below", restrict_fill_value="str"
    )
)
@pytest.mark.slow
def test_extrapolate_below_without_fill_value_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x_new is below the minimum of x and
    # fill_value=="extrapolate", which means constant extrapolation from the boundaries.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works, meaning in the present case, that the boundary
    # value of y is taken for all x_new below the original bound.
    assert np.all(
        y_new[interp_inputs["x_new"] < np.min(interp_inputs["x"])]
        == interp_inputs["y"][0]
    )


@given(values_uncertainties_kind(extrapolate="below", restrict_fill_value="float"))
@pytest.mark.slow
def test_extrapolate_below_with_fill_value_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x_new is below the minimum of x and
    # fill_value is a float, which means constant extrapolation with this value.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works.
    assert np.all(
        y_new[interp_inputs["x_new"] < np.min(interp_inputs["x"])]
        == interp_inputs["fill_value"]
    )


@given(values_uncertainties_kind(extrapolate="below", restrict_fill_value="tuple"))
@pytest.mark.slow
def test_extrapolate_below_with_fill_values_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x_new is below the minimum of x and
    # fill_value is a tuple, which means constant extrapolation with its first
    # element.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works.
    assert np.all(
        y_new[interp_inputs["x_new"] < np.min(interp_inputs["x"])]
        == interp_inputs["fill_value"][0]
    )


@given(
    values_uncertainties_kind(
        sorted_xs=True, extrapolate="above", restrict_fill_value="str"
    )
)
@pytest.mark.slow
def test_extrapolate_above_without_fill_value_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x_new is above the maximum of x and
    # fill_value=="extrapolate", which means constant extrapolation from the boundaries.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works, meaning in the present case, that the boundary
    # value of y is taken for all x _newabove the original bound.
    assert np.all(
        y_new[interp_inputs["x_new"] > np.max(interp_inputs["x"])]
        == interp_inputs["y"][-1]
    )


@given(values_uncertainties_kind(extrapolate="above", restrict_fill_value="float"))
@pytest.mark.slow
def test_extrapolate_above_with_fill_value_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x_new is above the maximum of x and
    # fill_value is a float, which means constant extrapolation with this value.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works.
    assert np.all(
        y_new[interp_inputs["x_new"] > np.max(interp_inputs["x"])]
        == interp_inputs["fill_value"]
    )


@given(values_uncertainties_kind(extrapolate="above", restrict_fill_value="tuple"))
@pytest.mark.slow
def test_extrapolate_above_with_fill_values_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x_new is above the maximum of x and
    # fill_value is a tuple, which means constant extrapolation with its second element.
    y_new = interp1d_unc(**interp_inputs)[1]
    # Check that extrapolation works.
    assert np.all(
        y_new[interp_inputs["x_new"] > np.max(interp_inputs["x"])]
        == interp_inputs["fill_value"][1]
    )


@given(
    values_uncertainties_kind(
        sorted_xs=True, extrapolate="below", restrict_fill_unc="str"
    )
)
@pytest.mark.slow
def test_extrapolate_below_without_fill_unc_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x_new is below the minimum of x and
    # fill_unc=="extrapolate", which means constant extrapolation from the boundaries.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works, meaning in the present case, that the boundary
    # value of y is taken for all x _newbelow the original bound.
    assert np.all(
        uy_new[interp_inputs["x_new"] < np.min(interp_inputs["x"])]
        == interp_inputs["uy"][0]
    )


@given(values_uncertainties_kind(extrapolate="below", restrict_fill_unc="float"))
@pytest.mark.slow
def test_extrapolate_below_with_fill_unc_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x_new is below the minimum of x and
    # fill_unc is a float, which means constant extrapolation with this value.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works.
    assert np.all(
        uy_new[interp_inputs["x_new"] < np.min(interp_inputs["x"])]
        == interp_inputs["fill_unc"]
    )


@given(values_uncertainties_kind(extrapolate="below", restrict_fill_unc="tuple"))
@pytest.mark.slow
def test_extrapolate_below_with_fill_uncs_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x _newis below the minimum of x and
    # fill_unc is a tuple, which means constant extrapolation with its first element.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works.
    assert np.all(
        uy_new[interp_inputs["x_new"] < np.min(interp_inputs["x"])]
        == interp_inputs["fill_unc"][0]
    )


@given(
    values_uncertainties_kind(
        sorted_xs=True, extrapolate="above", restrict_fill_unc="str"
    )
)
@pytest.mark.slow
def test_extrapolate_above_without_fill_unc_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x _newis above the maximum of x and
    # fill_unc=="extrapolate", which means constant extrapolation from the boundaries.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works, meaning in the present case, that the boundary
    # value of y is taken for all x _newabove the original bound.
    assert np.all(
        uy_new[interp_inputs["x_new"] > np.max(interp_inputs["x"])]
        == interp_inputs["uy"][-1]
    )


@given(values_uncertainties_kind(extrapolate="above", restrict_fill_unc="float"))
@pytest.mark.slow
def test_extrapolate_above_with_fill_unc_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x _newis above the maximum of x and
    # fill_unc is a float, which means constant extrapolation with this value.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works.
    assert np.all(
        uy_new[interp_inputs["x_new"] > np.max(interp_inputs["x"])]
        == interp_inputs["fill_unc"]
    )


@given(values_uncertainties_kind(extrapolate="above", restrict_fill_unc="tuple"))
@pytest.mark.slow
def test_extrapolate_above_with_fill_uncs_interp1d_unc(interp_inputs):
    # Deal with those cases where at least one of x _newis above the maximum of x and
    # fill_unc is a tuple, which means constant extrapolation with its second element.
    uy_new = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation works.
    assert np.all(
        uy_new[interp_inputs["x_new"] > np.max(interp_inputs["x"])]
        == interp_inputs["fill_unc"][1]
    )


@given(values_uncertainties_kind(returnC=True, kind_tuple=("linear",)))
@pytest.mark.slow
def test_compare_returnc_interp1d_unc(interp_inputs):
    # Compare the uncertainties computed from the sensitivities inside the
    # interpolation range and directly.
    uy_new_with_sensitivities = interp1d_unc(**interp_inputs)[2]
    interp_inputs["returnC"] = False
    uy_new_without_sensitivities = interp1d_unc(**interp_inputs)[2]
    # Check that extrapolation results match up to machine epsilon.
    assert_allclose(uy_new_with_sensitivities, uy_new_without_sensitivities, rtol=9e-15)


@given(
    values_uncertainties_kind(returnC=True, extrapolate=True, kind_tuple=("linear",))
)
@pytest.mark.slow
def test_failing_returnc_with_extrapolation_interp1d_unc(interp_inputs):
    # Since we have not implemented these cases, for now we
    # check for exception being thrown.
    assume(not isinstance(interp_inputs["fill_unc"], str))
    with raises(NotImplementedError):
        interp1d_unc(**interp_inputs)


@given(
    values_uncertainties_kind(
        returnC=True,
        extrapolate=True,
        kind_tuple=("linear", "cubic"),
        restrict_fill_unc="str",
    )
)
@pytest.mark.slow
def test_returnc_with_extrapolation_interp1d_unc(interp_inputs):
    # Check if extrapolation with constant values outside interpolation range and
    # returning of sensitivities is callable.
    assert interp1d_unc(**interp_inputs)


@given(
    values_uncertainties_kind(
        returnC=True,
        extrapolate=True,
        kind_tuple=("linear", "cubic"),
        restrict_fill_unc="str",
        sorted_xs=True,
    )
)
@pytest.mark.slow
def test_returnc_with_extrapolation_check_below_bound_interp1d_unc(interp_inputs):
    # Check if extrapolation with constant values outside interpolation range and
    # returning sensitivities work as expected regarding extrapolation values
    # below original bound.
    uy_new, C = interp1d_unc(**interp_inputs)[2:]
    assert np.all(
        uy_new[interp_inputs["x_new"] < np.min(interp_inputs["x"])]
        == interp_inputs["uy"][0]
    )


@given(
    values_uncertainties_kind(
        returnC=True,
        extrapolate=True,
        kind_tuple=("linear", "cubic"),
        restrict_fill_unc="str",
        sorted_xs=True,
    )
)
@pytest.mark.slow
def test_returnc_with_extrapolation_check_uy_new_above_bound_interp1d_unc(
    interp_inputs,
):
    # Check if extrapolation with constant values outside interpolation range and
    # returning sensitivities work as expected regarding extrapolation values
    # above original bound.
    uy_new = interp1d_unc(**interp_inputs)[2]
    assert np.all(
        uy_new[interp_inputs["x_new"] > np.max(interp_inputs["x"])]
        == interp_inputs["uy"][-1]
    )


@given(
    values_uncertainties_kind(
        returnC=True,
        extrapolate=True,
        kind_tuple=("linear",),
        restrict_fill_unc="str",
    )
)
@pytest.mark.slow
def test_returnc_with_extrapolation_check_c_interp1d_unc(
    interp_inputs,
):
    # Check if sensitivity computation parallel to linear interpolation and
    # extrapolation with constant values works as expected regarding the shape and
    # content of the sensitivity matrix.
    C = interp1d_unc(**interp_inputs)[3]

    # Check that C has the right shape.
    assert C.shape == (len(interp_inputs["x_new"]), len(interp_inputs["x"]))

    # Find interpolation range because we reuse it.
    interp_range = (interp_inputs["x_new"] >= np.min(interp_inputs["x"])) | (
        interp_inputs["x_new"] <= np.max(interp_inputs["x"])
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
    assert_allclose(np.sum(C, 1), np.ones_like(interp_inputs["x_new"]))


@given(
    values_uncertainties_kind(
        returnC=True,
        kind_tuple=(
            "previous",
            "next",
            "nearest",
        ),
    )
)
@pytest.mark.slow
def test_value_error_for_returnc_interp1d_unc(interp_inputs):
    # Check erroneous calls with returnC and wrong kind.
    with raises(NotImplementedError):
        interp1d_unc(**interp_inputs)


@given(hst.integers(min_value=3, max_value=1000))
def test_linear_uy_in_interp1d_unc(
    n,
):
    # Check for given input, if interpolated uncertainties equal 1 and
    # :math:`sqrt(2) / 2`.
    dx_unit = 2
    dx_half = dx_unit / 2
    x_new = np.arange(0, n, dx_half)
    x_unit = np.arange(0, n + dx_half, dx_unit)
    y = uy_unit = np.ones_like(x_unit)
    uy_new = interp1d_unc(x_new, x_unit, y, uy_unit, "linear")[2]
    assert np.all(uy_new[0:n:dx_unit] == 1) and np.all(
        uy_new[1:n:dx_unit] == np.sqrt(2) / 2
    )


@given(values_uncertainties_kind())
@pytest.mark.slow
def test_wrong_input_lengths_call_interp1d(interp_inputs):
    # Check erroneous calls with unequally long inputs.
    with raises(ValueError):
        interp_inputs["uy"] = np.tile(interp_inputs["uy"], 2)
        interp1d_unc(**interp_inputs)


@given(values_uncertainties_kind(kind_tuple=("spline", "least-squares")))
@pytest.mark.slow
def test_raise_not_implemented_yet_interp1d(interp_inputs):
    # Check that not implemented versions raise exceptions.
    with raises(NotImplementedError):
        interp1d_unc(**interp_inputs)


@given(values_uncertainties_kind(extrapolate=True))
@pytest.mark.slow
def test_raise_value_error_interp1d_unc(interp_inputs):
    # Check that interpolation with points outside the original domain raises
    # exception if requested.
    interp_inputs["bounds_error"] = True
    with raises(ValueError):
        interp1d_unc(**interp_inputs)


# noinspection PyArgumentList
@given(values_uncertainties_kind(for_make_equidistant=True))
@pytest.mark.slow
def test_too_short_call_make_equidistant(interp_inputs):
    # Check erroneous calls with too few inputs.
    with raises(TypeError):
        make_equidistant(interp_inputs["x"])
        make_equidistant(interp_inputs["x"], interp_inputs["y"])


@given(values_uncertainties_kind(for_make_equidistant=True))
@pytest.mark.slow
def test_full_call_make_equidistant(interp_inputs):
    t_new, y_new, uy_new = make_equidistant(**interp_inputs)
    # Check the equal dimensions of the minimum calls output.
    assert len(t_new) == len(y_new) == len(uy_new)


@given(values_uncertainties_kind(for_make_equidistant=True))
@pytest.mark.slow
def test_wrong_input_lengths_call_make_equidistant(interp_inputs):
    # Check erroneous calls with unequally long inputs.
    with raises(ValueError):
        y_wrong = np.tile(interp_inputs["y"], 2)
        uy_wrong = np.tile(interp_inputs["uy"], 3)
        make_equidistant(interp_inputs["x"], y_wrong, uy_wrong)


@given(values_uncertainties_kind(for_make_equidistant=True))
@pytest.mark.slow
def test_t_new_to_dt_make_equidistant(interp_inputs):
    x_new = make_equidistant(**interp_inputs)[0]
    delta_x_new = np.diff(x_new)
    # Check if x_new is ascending.
    assert not np.any(delta_x_new < 0)


@given(
    values_uncertainties_kind(
        kind_tuple=("previous", "next", "nearest"), for_make_equidistant=True
    )
)
@pytest.mark.slow
def test_prev_in_make_equidistant(interp_inputs):
    y_new, uy_new = make_equidistant(**interp_inputs)[1:3]
    # Check if all 'interpolated' values are present in the actual values.
    assert np.all(np.isin(y_new, interp_inputs["y"]))
    assert np.all(np.isin(uy_new, interp_inputs["uy"]))


@given(values_uncertainties_kind(kind_tuple=["linear"], for_make_equidistant=True))
@pytest.mark.slow
def test_linear_in_make_equidistant(interp_inputs):
    y_new, uy_new = make_equidistant(**interp_inputs)[1:3]
    # Check if all interpolated values lie in the range of the original values.
    assert np.all(np.amin(interp_inputs["y"]) <= y_new)
    assert np.all(np.amax(interp_inputs["y"]) >= y_new)


@given(hst.integers(min_value=3, max_value=1000))
@pytest.mark.slow
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


@given(
    values_uncertainties_kind(
        kind_tuple=("spline", "least-squares"), for_make_equidistant=True
    )
)
@pytest.mark.slow
def test_raise_not_implemented_yet_make_equidistant(interp_inputs):
    # Check that not implemented versions raise exceptions.
    with raises(NotImplementedError):
        make_equidistant(**interp_inputs)
