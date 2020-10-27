"""Perform a speed comparison for uncertainty.propagate_filter.FIRuncFilter
 with different value-types of the covariance of theta
"""

import numpy as np
import time
import pandas as pd

from PyDynamic.uncertainty.propagate_filter import FIRuncFilter

def evaluate(signal, filter, case="full/null/none"):
    # different covariance matrices
    if case == "full":
        U_theta = 1e-2 * np.eye(filter_length)
    elif case == "null":
        U_theta = np.zeros((filter_length, filter_length))
    elif case == "none":
        U_theta = None
    else:
        raise NotImplementedError
    
    # measure runtime (timeit.timeit is unable to use both local+global variables)
    start = time.time()
    y, Uy = FIRuncFilter(signal, sigma_noise, theta, U_theta, blow=None)
    end = time.time()
    runtime = end - start

    return runtime, y, Uy


filter_lengths = [10, 50, 100, 200, 500, 1000, 2000]
cases = ["full", "null", "none"]
results = pd.DataFrame(index=filter_lengths, columns=cases)


# prepare signal
signal_length = 10000
sigma_noise = 1e-2  # 1e-5
signal = sigma_noise * np.random.randn(signal_length)

for filter_length in filter_lengths:

    # prepare filter parameters
    theta = np.random.random(filter_length)

    for case in cases:
        runtime, y, uy = evaluate(signal, theta, case)
        results.loc[filter_length, case] = runtime

        print(signal_length, filter_length, case, uy[:2])
    print("="*10)

# The expected result is, that `full` and `null` should be equally fast.
# `none` will be much faster, although it yields the same results as `null`.
print(results)
