# -*- coding: utf-8 -*-
""" Perform tests on the method *make_equidistant*."""

import numpy as np
from pytest import raises

from PyDynamic.misc.tools import make_equidistant

n = 50
t, y, uy, dt, kind = np.linspace(0, 1, n), np.random.uniform(size=n), \
                     np.random.uniform(size=n), 5e-2, 'previous'


def test_too_short_call_make_equidistant():
    # Check erroneous calls with too few inputs.
    with raises(TypeError):
        make_equidistant(t)
        make_equidistant(t, y)


def test_wrong_input_lengths_call_make_equidistant():
    # Check erroneous calls with unequally long inputs.
    with raises(ValueError):
        y_n_wrong = n * 2
        uy_n_wrong = n * 3
        y_wrong, uy_wrong = np.random.uniform(size=y_n_wrong), \
                         np.random.uniform(size=uy_n_wrong)
        make_equidistant(t, y_wrong, uy_wrong)


def test_wrong_input_order_call_make_equidistant():
    # Check erroneous calls with not ascending timestamps.
    with raises(ValueError):
        # Assure first and last value are correctly ordered and COUNT matches.
        t_wrong = np.empty_like(t)
        t_wrong[0] = t[0]
        t_wrong[-1] = t[-1]
        t_wrong[1:-1] = -np.sort(-t[1:-1])
        make_equidistant(t_wrong, y, uy)


def test_minimal_call_make_equidistant():
    # Check the minimum working call.
    t_new, y_new, uy_new = make_equidistant(t, y, uy)
    # Check the equal dimensions of the minimum calls output.
    assert len(t_new) == len(y_new) == len(uy_new)


def test_full_call_make_equidistant():
    # Setup array of timesteps in relation to original time interval length.
    dts = np.power(10., np.arange(-5, 0)) * (t[-1] - t[0])
    # Check full call for all specified timesteps.
    for i_dt in dts:
        make_equidistant(t, y, uy, dt=i_dt)
    # Setup array of all implemented interpolation methods.
    kinds = ['previous', 'next', 'nearest', 'linear']
    # Check all possible full interpolation calls.
    for i_kind in kinds:
        make_equidistant(t, y, uy, kind=i_kind)
        for i_dt in dts:
            make_equidistant(t, y, uy, i_dt, i_kind)
            make_equidistant(t, y, uy, dt=i_dt, kind=i_kind)


def test_t_new_to_dt_make_equidistant():
    from numpy.testing import assert_almost_equal

    t_new = make_equidistant(t, y, uy, dt)[0]
    delta_t_new = np.diff(t_new)
    # Check if the new timestamps are ascending.
    assert np.all(delta_t_new > 0)
    # Check if the timesteps have the desired length.
    assert_almost_equal(delta_t_new - dt, 0)


def test_prev_in_make_equidistant():
    kinds = ['previous', 'next', 'nearest']
    for i_kind in kinds:
        y_new, uy_new = make_equidistant(t, y, uy, dt, i_kind)[1:3]
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


def test_linear_uy_in_make_equidistant():
    # Check for given input, if interpolated uncertainties equal 1 and
    # :math:`sqrt(2) / 2`.
    dt_unit = 2
    t_unit = np.arange(0, n, dt_unit)
    y = np.ones_like(t_unit)
    uy_unit = np.ones_like(t_unit)
    dt_half = 1
    uy_new = make_equidistant(t_unit, y, uy_unit, dt_half, 'linear')[2]
    assert np.all(uy_new[0:n:2] == 1) and np.all(uy_new[1:n:2] == np.sqrt(2)
                                                 / 2)


def test_raise_not_implemented_yet_make_equidistant():
    # Check that not implemented versions raise exceptions.
    with raises(NotImplementedError):
        make_equidistant(t, y, uy, dt, 'spline')
        make_equidistant(t, y, uy, dt, 'lagrange')
        make_equidistant(t, y, uy, dt, 'least-squares')
