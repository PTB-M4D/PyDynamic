from typing import Dict, Optional, Tuple, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
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
            "spline", "lagrange", "least-squares" from which the strategy for the
            kind randomly chooses. Defaults to the valid options "linear",
            "previous", "next", "nearest".
        sorted_timestamps: bool
            if the timestamps should be in ascending order or not

    Returns
    -------
        A dict containing the randomly generated expected input parameters t, y, uy,
        dt, kind for make_equidistant()
    """
    # Set all common parameters for timestamps, measurements values and associated
    # uncertainties.
    shape_for_timestamps = hnp.array_shapes(
        max_dims=1, min_side=min_count, max_side=max_count
    )
    strategy_params = {
        "dtype": np.float,
        "shape": shape_for_timestamps,
        "elements": st.floats(
            min_value=0, max_value=1e300, allow_nan=False, allow_infinity=False
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
    # Reset shape for interpolation timestamps and use range of "original" timestamps
    # as boundaries.
    strategy_params["shape"] = shape_for_timestamps
    strategy_params["elements"] = st.floats(
        min_value=np.min(t), max_value=np.max(t), allow_nan=False, allow_infinity=False
    )
    t_new = draw(hnp.arrays(**strategy_params))
    kind = draw(st.sampled_from(kind_tuple))
    return {"t_new": t_new, "t": t, "y": y, "uy": uy, "kind": kind}


@given(timestamps_values_uncertainties_kind())
def test_usual_call(interp_inputs):
    assert interp1d_unc(**interp_inputs) is not None


@given(timestamps_values_uncertainties_kind(min_count=1, max_count=1))
def test_too_few_timestamps_call(interp_inputs):
    # Check that too few timestamps raise exceptions.
    with raises(ValueError):
        interp1d_unc(**interp_inputs)


@given(timestamps_values_uncertainties())
def test_raise_not_implemented_yet_interp1d(interp_inputs):
    # Check that not implemented versions raise exceptions.
    with raises(NotImplementedError):
        interp1d_unc(**interp_inputs)
