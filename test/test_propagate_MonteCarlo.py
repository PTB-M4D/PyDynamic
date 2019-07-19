# -*- coding: utf-8 -*-
""" Perform tests on the method *make_equidistant*."""

import numpy as np
from pytest import raises

import sys
sys.path.append(".")

import PyDynamic.uncertainty.propagate_MonteCarlo as mc


def test_Normal_ZeroCorr_constructor():
    N = 10
    possibleInputs = [None, 
           0, 0.0, np.zeros(N), np.zeros(N+1), 
           1, 1.0, np.ones(N), np.ones(N+1), 
           np.arange(N), np.arange(N+1)]
    
    for loc in possibleInputs:
        for scale in possibleInputs:
            if isinstance(loc, np.ndarray) or isinstance(scale, np.ndarray):

                if isinstance(loc, np.ndarray) and isinstance(scale, np.ndarray):
                    if len(loc) != len(scale):
                        with raises(AssertionError):
                            zc = mc.Normal_ZeroCorr(loc=loc, scale=scale)
                        continue
                    else:
                        zc = mc.Normal_ZeroCorr(loc=loc, scale=scale)
                else:
                    zc = mc.Normal_ZeroCorr(loc=loc, scale=scale)

            else:
                with raises(TypeError):
                    zc = mc.Normal_ZeroCorr(loc=loc, scale=scale)
                continue
            
            # run the rvs method
            Nmax = max(len(zc.mean), len(zc.std))
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
