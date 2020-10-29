import inspect

import pytest

from PyDynamic import uncertainty


@pytest.mark.parametrize(
    "function", [uncertainty.interpolation.interp1d_unc],
)
def test_deprecated_call(function):
    module = inspect.getmodule(function)
    pytest.deprecated_call(function)
    pytest.deprecated_call(getattr(module, function.__name__))
