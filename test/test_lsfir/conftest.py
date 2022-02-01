import os
import pathlib
from typing import Callable, Optional, Union

import hypothesis.strategies as hst
import numpy as np
import pytest
import scipy.signal as dsp
from hypothesis.strategies import composite
from numpy.testing import assert_allclose, assert_almost_equal

from PyDynamic.misc.SecondOrderSystem import sos_phys2filter
from PyDynamic.misc.tools import (
    complex_2_real_imag,
    make_semiposdef,
    real_imag_2_complex,
)
from ..conftest import (
    hypothesis_float_vector,
    scale_matrix_or_vector_to_convex_combination,
)


@pytest.fixture(scope="package")
def reference_array_path():
    return os.path.join(pathlib.Path(__file__).parent.resolve(), "reference_arrays")


@pytest.fixture(scope="package")
def random_number_generator():
    return np.random.default_rng(1)


@pytest.fixture(scope="package")
def measurement_system():
    f0 = 36e3
    S0 = 0.4
    delta = 0.01
    return {"f0": f0, "S0": S0, "delta": delta}


@composite
def weights(
    draw: Callable, guarantee_vector: Optional[bool] = False
) -> Union[np.ndarray, None]:
    valid_vector_strategy = hypothesis_float_vector(
        length=400, min_value=0, max_value=1, exclude_min=True
    )
    valid_weight_strategies = (
        valid_vector_strategy
        if guarantee_vector
        else (valid_vector_strategy, hst.just(None))
    )
    unscaled_weights = draw(hst.one_of(valid_weight_strategies))
    if unscaled_weights is not None:
        return scale_matrix_or_vector_to_convex_combination(unscaled_weights)


@pytest.fixture(scope="package")
def sampling_freq():
    return 500e3


@pytest.fixture(scope="package")
def freqs():
    return np.linspace(0, 120e3, 200)


@pytest.fixture(scope="package")
def monte_carlo(
    measurement_system,
    random_number_generator,
    sampling_freq,
    freqs,
    complex_freq_resp,
    reference_array_path,
):
    udelta = 0.1 * measurement_system["delta"]
    uS0 = 0.001 * measurement_system["S0"]
    uf0 = 0.01 * measurement_system["f0"]

    runs = 10000
    MCS0 = random_number_generator.normal(
        loc=measurement_system["S0"], scale=uS0, size=runs
    )
    MCd = random_number_generator.normal(
        loc=measurement_system["delta"], scale=udelta, size=runs
    )
    MCf0 = random_number_generator.normal(
        loc=measurement_system["f0"], scale=uf0, size=runs
    )
    HMC = np.empty((runs, len(freqs)), dtype=complex)
    for index, mcs0_mcd_mcf0 in enumerate(zip(MCS0, MCd, MCf0)):
        bc_, ac_ = sos_phys2filter(mcs0_mcd_mcf0[0], mcs0_mcd_mcf0[1], mcs0_mcd_mcf0[2])
        b_, a_ = dsp.bilinear(bc_, ac_, sampling_freq)
        HMC[index, :] = dsp.freqz(b_, a_, 2 * np.pi * freqs / sampling_freq)[1]

    H = complex_2_real_imag(complex_freq_resp)
    assert_allclose(
        H,
        np.load(
            os.path.join(reference_array_path, "test_LSFIR_H.npz"),
        )["H"],
    )
    uAbs = np.std(np.abs(HMC), axis=0)
    assert_allclose(
        uAbs,
        np.load(
            os.path.join(reference_array_path, "test_LSFIR_uAbs.npz"),
        )["uAbs"],
        rtol=3.5e-2,
    )
    uPhas = np.std(np.angle(HMC), axis=0)
    assert_allclose(
        uPhas,
        np.load(
            os.path.join(reference_array_path, "test_LSFIR_uPhas.npz"),
        )["uPhas"],
        rtol=4.3e-2,
    )
    UH = np.cov(np.hstack((np.real(HMC), np.imag(HMC))), rowvar=False)
    UH = make_semiposdef(UH)
    assert_allclose(
        UH,
        np.load(
            os.path.join(reference_array_path, "test_LSFIR_UH.npz"),
        )["UH"],
        atol=1,
    )
    return {"H": H, "uAbs": uAbs, "uPhas": uPhas, "UH": UH}


@pytest.fixture(scope="package")
def complex_H_with_UH(monte_carlo):
    return {
        "H": real_imag_2_complex(monte_carlo["H"]),
        "UH": monte_carlo["UH"],
    }


@pytest.fixture(scope="package")
def digital_filter(measurement_system, sampling_freq):
    # transform continuous system to digital filter
    bc, ac = sos_phys2filter(
        measurement_system["S0"], measurement_system["delta"], measurement_system["f0"]
    )
    assert_almost_equal(bc, [20465611686.098896])
    assert_allclose(ac, np.array([1.00000000e00, 4.52389342e03, 5.11640292e10]))
    b, a = dsp.bilinear(bc, ac, sampling_freq)
    assert_allclose(
        b, np.array([0.019386043211510096, 0.03877208642302019, 0.019386043211510096])
    )
    assert_allclose(a, np.array([1.0, -1.7975690550957188, 0.9914294872108197]))
    return {"b": b, "a": a}


@pytest.fixture(scope="package")
def complex_freq_resp(
    measurement_system, sampling_freq, freqs, digital_filter, reference_array_path
):
    Hf = dsp.freqz(
        digital_filter["b"],
        digital_filter["a"],
        2 * np.pi * freqs / sampling_freq,
    )[1]
    assert_allclose(
        Hf,
        np.load(
            os.path.join(reference_array_path, "test_LSFIR_Hf.npz"),
        )["Hf"],
    )
    return Hf
