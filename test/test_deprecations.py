import inspect

import pytest

from PyDynamic import deconvolution, identification


@pytest.mark.parametrize(
    "function",
    [
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
    pytest.deprecated_call(function)
    pytest.deprecated_call(getattr(module, function.__name__))
