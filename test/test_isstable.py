# -*- coding: utf-8 -*-
""" Perform tests on *misc.filterstuff.isstable*."""

import numpy as np
import pytest

from PyDynamic.misc.filterstuff import isstable


def test_stable():
    # Design a filter, which is known to be stable.
    b = np.array([1, 1])
    a = np.array([1, -0.999999999])
    assert isstable(b, a)


def test_not_stable():
    # Design a filter, which is known to be unstable.
    b = np.array([1, 1])
    a = np.array([-0.999999999, 1])
    assert not isstable(b, a)


def test_error_isstable():
    """Test if isstable raises an error for invalid ftype"""
    b = np.array([1, 1])
    a = np.array([-0.999999999, 1])
    with pytest.raises(ValueError):
        isstable(b, a, ftype="something_invalid")
