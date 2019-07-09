# -*- coding: utf-8 -*-
""" Perform tests on the method *make_equidistant*."""

import numpy as np
from pytest import raises, approx

from PyDynamic.misc.tools import make_equidistant

n = 20
t, y, uy, dt, kind = np.linspace(0, 1, n), np.random.uniform(size=n), \
                     np.random.uniform(size=n), 5e-2, 'previous'


def test_too_short_call_make_equidistant():
    with raises(TypeError):
        make_equidistant(t)
        make_equidistant(t, y)


def test_minimal_call_make_equidistant():
    t_new, y_new = make_equidistant(t, y, uy)[0:2]
    assert len(t_new) == len(y_new)


def test_full_call_make_equidistant():
    make_equidistant(t, y, uy, dt, kind)
    make_equidistant(t, y, uy, dt=.5, kind='previous')
    make_equidistant(t, y, uy, dt=.5)
    make_equidistant(t, y, uy, kind='previous')


def test_t_new_to_dt_make_equidistant():
    t_new = make_equidistant(t, y, uy, dt, kind)[0]
    assert np.diff(t_new) - dt == approx(0)


def test_prev_in_make_equidistant():
    y_new, uy_new = make_equidistant(t, y, uy, dt, kind)[1:3]
    assert (np.isin(y_new, y)).all()
    assert (np.isin(uy_new, uy)).all()


def test_raise_not_implemented_yet_make_equidistant():
    with raises(NotImplementedError):
        make_equidistant(t, y, uy, dt, 'linear')
