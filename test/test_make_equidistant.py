# -*- coding: utf-8 -*-
""" Perform tests on the method *make_equidistant*."""

import numpy as np
from pytest import raises

from PyDynamic.misc.tools import make_equidistant

n = 20
t, y, uy, dt, kind = np.linspace(0, 1, n), np.random.uniform(size=n), \
                     np.random.uniform(size=n), 5e-2, 'previous'


def test_too_short_call_make_equidistant():
    # Check erroneous calls.
    with raises(TypeError):
        make_equidistant(t)
        make_equidistant(t, y)


def test_minimal_call_make_equidistant():
    # Check the minimum working call.
    t_new, y_new, uy_new = make_equidistant(t, y, uy)
    # Check the equal dimensions of the minimum calls output.
    assert len(t_new) == len(y_new) == len(uy)


def test_full_call_make_equidistant():
    # Check all possible working calls.
    make_equidistant(t, y, uy, dt, kind)
    make_equidistant(t, y, uy, dt, 'linear')
    make_equidistant(t, y, uy, dt=.5, kind='previous')
    make_equidistant(t, y, uy, dt=.5, kind='linear')
    make_equidistant(t, y, uy, dt=.5)
    make_equidistant(t, y, uy, kind='previous')
    make_equidistant(t, y, uy, kind='linear')


def test_t_new_to_dt_make_equidistant():
    from numpy.testing import assert_almost_equal

    t_new = make_equidistant(t, y, uy, dt, kind)[0]
    difference = np.diff(t_new)
    # Check if the new timestamps are ascending.
    assert np.all(difference > 0)
    # Check if the timesteps have the desired length.
    assert_almost_equal(difference - dt, 0)


def test_prev_in_make_equidistant():
    y_new, uy_new = make_equidistant(t, y, uy, dt, kind)[1:3]
    # Check if all 'interpolated' values are present in the actual values.
    assert np.all(np.isin(y_new, y))
    assert np.all(np.isin(uy_new, uy))


def test_linear_in_make_equidistant():
    y_new, uy_new = make_equidistant(t, y, uy, dt, 'linear')[1:3]
    # Check if all interpolated values lie in the range of the actual values.
    assert np.all(np.amin(y) <= y_new)
    assert np.all(np.amax(y) >= y_new)
    # Check if for any non-zero input uncertainty the output contains non-zeros.
    if np.any(uy):
        assert np.any(uy_new)


def test_raise_not_implemented_yet_make_equidistant():
    # Check that not implemented versions raise exceptions.
    with raises(NotImplementedError):
        make_equidistant(t, y, uy, dt, 'spline')
        make_equidistant(t, y, uy, dt, 'lagrange')
        make_equidistant(t, y, uy, dt, 'least-squares')
