import numpy as np
import pytest
from hypothesis import given, HealthCheck, settings
from numpy.testing import assert_allclose
from PyDynamic import FIRuncFilter, MC

from .conftest import _set_irrelevant_ranges_to_zero
from ..conftest import FIRuncFilter_input


@given(FIRuncFilter_input(exclude_corr_kind=True))
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.too_slow,
    ],
    max_examples=50,
)
@pytest.mark.slow
def test(fir_unc_filter_input):
    # In this test, we exclude the case of a valid signal with uncertainty given as
    # the right-sided auto-covariance (acf). This is done, because we currently do not
    # ensure, that the random-drawn acf generates a positive-semidefinite
    # Toeplitz-matrix. Therefore we cannot construct a valid and equivalent input for
    # the Monte-Carlo method in that case.

    # Check output for thinkable permutations of input parameters against a Monte Carlo
    # approach.

    # run method
    y_fir, Uy_fir = FIRuncFilter(**fir_unc_filter_input, return_full_covariance=True)

    # run Monte Carlo simulation of an FIR
    # adjust input to match conventions of MC
    x = fir_unc_filter_input["y"]
    ux = fir_unc_filter_input["sigma_noise"]

    b = fir_unc_filter_input["theta"]
    a = np.ones(1)
    if fir_unc_filter_input["Utheta"] is None:
        Uab = np.zeros((len(b), len(b)))
    else:
        Uab = fir_unc_filter_input["Utheta"]

    blow = fir_unc_filter_input["blow"]
    if isinstance(blow, np.ndarray):
        n_blow = len(blow)
    else:
        n_blow = 0

    # run FIR with MC and extract diagonal of returned covariance
    y_mc, Uy_mc = MC(
        x,
        ux,
        b,
        a,
        Uab,
        blow=blow,
        runs=2000,
        shift=-fir_unc_filter_input["shift"],
        verbose=True,
    )

    # approximate comparison after swing-in of MC-result (which is after the combined
    # length of blow and b)
    swing_in_length = len(b) + n_blow
    relevant_y_fir, relevant_Uy_fir = _set_irrelevant_ranges_to_zero(
        signal=y_fir,
        uncertainties=Uy_fir,
        swing_in_length=swing_in_length,
        shift=fir_unc_filter_input["shift"],
    )
    relevant_y_mc, relevant_Uy_mc = _set_irrelevant_ranges_to_zero(
        signal=y_mc,
        uncertainties=Uy_mc,
        swing_in_length=swing_in_length,
        shift=fir_unc_filter_input["shift"],
    )

    # HACK for visualization during debugging
    # from PyDynamic.misc.tools import plot_vectors_and_covariances_comparison
    #
    # plot_vectors_and_covariances_comparison(
    #     vector_1=relevant_y_fir,
    #     vector_2=relevant_y_mc,
    #     covariance_1=relevant_Uy_fir,
    #     covariance_2=relevant_Uy_mc,
    #     label_1="fir",
    #     label_2="mc",
    #     title=f"filter length: {len(b)}, signal length: {len(x)}, blow: "
    #     f"{fir_unc_filter_input['blow']}",
    # )
    # /HACK
    assert_allclose(
        relevant_y_fir,
        relevant_y_mc,
        atol=np.max((2 * np.max(np.abs(y_fir)), 2e-1)),
    )
    assert_allclose(
        relevant_Uy_fir,
        relevant_Uy_mc,
        atol=np.max((2 * np.max(Uy_fir), 1e-7)),
    )
