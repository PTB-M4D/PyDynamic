# -*- coding: utf-8 -*-
""" Perform tests on the method *uncertainty.propagate_MonteCarlo*"""

import numpy as np
from pytest import raises

import PyDynamic.uncertainty.propagate_MonteCarlo as mc


N = 10
possibleInputs = [0, 0.0, np.zeros(1), np.zeros(N), np.zeros(N+1),
                    1, 1.0, np.ones(1), np.ones(N), np.ones(N+1),
                    np.arange(N), np.arange(N+1)]


def shouldRaiseTypeError(a, b):
    return not isinstance(a, np.ndarray) and not isinstance(b, np.ndarray)


def shouldRaiseValueError(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.size != b.size:
            return (not len(a) == 1) and (not len(b) == 1)
        else:
            return False
    else:
        return False


def test_Normal_ZeroCorr_constructor():
    
    for loc in possibleInputs:
        for scale in possibleInputs:

            if shouldRaiseTypeError(loc, scale):
                with raises(TypeError):
                    zc = mc.Normal_ZeroCorr(loc=loc, scale=scale)
            
            elif shouldRaiseValueError(loc, scale):
                with raises(ValueError):
                    zc = mc.Normal_ZeroCorr(loc=loc, scale=scale)

            else: # should run through
                zc = mc.Normal_ZeroCorr(loc=loc, scale=scale)

                # run the rvs method
                Nmax = max(zc.loc.size, zc.scale.size)
                k = np.random.choice(np.arange(Nmax), size=1)[0]
                rvs = zc.rvs(size=k)

                # check assertions
                assert isinstance(rvs, np.ndarray)
                assert rvs.shape == (k, Nmax)


def test_MC():
    # maybe take this test from some example?
    pass


def test_SMC():
    # maybe take this test from some example?
    pass


def test_UMC():
    # maybe take this test from some example?
    pass
