import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis.strategies import composite

from PyDynamic.uncertainty.interpolation import interp1d_unc


@composite
def timestamps_values_uncertainties(draw, min_count):

    strategy_params = {
        "dtype": np.float,
        "shape": hnp.array_shapes(max_dims=1, min_side=min_count),
        "elements": st.floats(min_value=0, allow_nan=False, allow_infinity=False),
        "unique": True,
    }
    t = draw(hnp.arrays(**strategy_params))
    t.sort()
    strategy_params["shape"] = np.shape(t)
    y = draw(hnp.arrays(**strategy_params))
    uy = draw(hnp.arrays(**strategy_params))
    strategy_params["elements"] = st.floats(
        min_value=np.min(t), max_value=np.max(t), allow_nan=False, allow_infinity=False
    )
    t_new = draw(hnp.arrays(**strategy_params))
    t_new.sort()
    return {"t_new": t_new, "t": t, "y": y, "uy": uy}


@given(timestamps_values_uncertainties(min_count=2))
def test_minimal_call(interp_inputs):

    interpolation = interp1d_unc(**interp_inputs)
    assert interpolation is not None
