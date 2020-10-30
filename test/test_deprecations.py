import inspect

import pytest

from PyDynamic import deconvolution, identification
from PyDynamic.misc.tools import make_equidistant
from PyDynamic.uncertainty.interpolation import interp1d_unc


@pytest.mark.parametrize(
    "function",
    [
        interp1d_unc,
        make_equidistant,
        deconvolution.LSIIR,
        deconvolution.LSFIR,
        deconvolution.LSFIR_unc,
        deconvolution.LSFIR_uncMC,
        deconvolution.LSIIR_unc,
        identification.LSIIR,
        identification.LSFIR,
        identification.fit_sos,
    ],
)
def test_deprecated_call(function):
    module = inspect.getmodule(function)
    with pytest.raises(DeprecationWarning):
        function()
    with pytest.raises(DeprecationWarning):
        getattr(module, function.__name__).__call__()
