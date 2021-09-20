"""Perform a speed comparison for uncertainty.propagate_filter.FIRuncFilter
 with different value-types of the covariance of theta
"""

import numpy as np
import time
import pandas as pd

from PyDynamic.uncertainty.propagate_filter import FIRuncFilter

def evaluate(signal, filter, case="full/zero/none"):
    # different covariance matrices
    if case == "full":
        U_theta = 1e-2 * np.eye(filter_length)
    elif case == "zero":
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


filter_lengths = [10, 50, 100, 200, 500, 1000]
cases = ["full", "zero", "none"]
results = pd.DataFrame(index=filter_lengths, columns=cases)


# prepare signal
signal_length = 2000
sigma_noise = 1e-2  # 1e-5
signal = sigma_noise * np.random.randn(signal_length)

# print table header
table_hdr = "{0:<15s} | {1:<15s} | {2:<10s} | {3:<15s}"
table_fmt = "{0:<15d} | {1:<15d} | {2:<10s} | {3}"
print(table_hdr.format("signal-length", "filter-length", "case", "uy[:2]"))
print("-"*68)

for filter_length in filter_lengths:

    # prepare filter parameters
    theta = np.random.random(filter_length)
    
    for case in cases:
        runtime, y, uy = evaluate(signal, theta, case)
        results.loc[filter_length, case] = runtime
        print(table_fmt.format(signal_length, filter_length, case, uy[:2]))
    print("-"*68)

# The expected result is, that `full` and `zero` should be roughly as fast (same range/magnitude).
# `none` will be much faster and yields the same results as `zero`.
print("\n\nThese are the observed execution times (in seconds) for \ndifferent combinations of filter-lengths and `case`:")
print(results)
