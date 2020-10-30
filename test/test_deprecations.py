import inspect

import pytest

from PyDynamic import uncertainty
from PyDynamic.misc.tools import make_equidistant


@pytest.mark.parametrize(
    "function", [uncertainty.interpolation.interp1d_unc, make_equidistant],
)
def test_deprecated_call(function):
    module = inspect.getmodule(function)
    with pytest.raises(DeprecationWarning):
        function()
    with pytest.raises(DeprecationWarning):
        getattr(module, function.__name__).__call__()
