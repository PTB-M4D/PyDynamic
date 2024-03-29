import hypothesis.strategies as hst
import pytest
from hypothesis import given, settings

# noinspection PyProtectedMember
from PyDynamic.misc.tools import is_2d_square_matrix, number_of_rows_equals_vector_dim
from PyDynamic.model_estimation.fit_filter import (
    LSFIR,
)
from .conftest import weights
from ..conftest import (
    _is_np_array,
    hypothesis_dimension,
)


@given(
    hypothesis_dimension(min_value=4, max_value=8),
    weights(),
    hst.booleans(),
    hst.booleans(),
)
@settings(deadline=None)
@pytest.mark.slow
def test(monte_carlo, freqs, sampling_freq, filter_order, weight_vector, verbose, inv):
    b, Ub = LSFIR(
        H=monte_carlo["H"],
        N=filter_order,
        f=freqs,
        Fs=sampling_freq,
        tau=filter_order // 2,
        weights=weight_vector,
        verbose=verbose,
        inv=inv,
        UH=monte_carlo["UH"],
        mc_runs=2,
    )
    assert _is_np_array(b) and len(b) == filter_order + 1
    assert is_2d_square_matrix(Ub) and number_of_rows_equals_vector_dim(Ub, b)
