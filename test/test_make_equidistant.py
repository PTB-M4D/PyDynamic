# -*- coding: utf-8 -*-
"""
Perform tests on the method *make_equidistant*.
"""

import numpy as np
import pytest

from PyDynamic.misc.tools import make_equidistant

n = 20
t, y, uy, dt, kind = 20, np.linspace(0, 1, n), np.random.uniform(size=n),\
                     np.random.uniform(), 'previous'


def test_too_short_call_make_equidistant():
    with pytest.raises(TypeError):
        make_equidistant(t)
        make_equidistant(t, y)


def test_minimal_call_make_equidistant():
    make_equidistant(t, y, uy)


def test_full_call_make_equidistant():
    make_equidistant(t, y, uy, dt, kind)
