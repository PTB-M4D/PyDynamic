"""Perform a speed comparison for :func:`PyDynamic.FIRuncFilter"""

import time

import numpy as np
import pandas as pd
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter


def evaluate():
    if covariance_case == "full":
        U_theta = np.diag(np.full_like(theta, sigma_noise))
    elif covariance_case == "zero":
        len_theta = len(theta)
        U_theta = np.zeros((len_theta, len_theta))
    elif covariance_case == "none":
        U_theta = None
    else:
        raise NotImplementedError

    # measure runtime (timeit.timeit is unable to use both local+global variables)
    start_time = time.time()
    current_y, current_Uy = FIRuncFilter(signal, sigma_noise, theta, U_theta, blow=None)
    end_time = time.time()

    return end_time - start_time, current_y, current_Uy


filter_lengths = [10, 50, 100, 200, 500, 1000]
covariance_cases = ["full", "zero", "none"]
results = pd.DataFrame(index=filter_lengths, columns=covariance_cases)


# prepare signal
signal_length = 2000
sigma_noise = 1e-2
signal = sigma_noise * np.random.randn(signal_length)

# print table header
table_hdr = "{0:<15s} | {1:<15s} | {2:<10s} | {3:<15s}"
table_fmt = "{0:<15d} | {1:<15d} | {2:<10s} | {3}"
print(table_hdr.format("signal-length", "filter-length", "case", "uy[:2]"))
print("-" * 68)

for filter_length in filter_lengths:

    # prepare filter parameters
    theta = np.random.random(filter_length)

    for covariance_case in covariance_cases:
        elapsed_time, y, uy = evaluate()
        results.loc[filter_length, covariance_case] = elapsed_time
        print(table_fmt.format(signal_length, filter_length, covariance_case, uy[:2]))
    print("-" * 68)

# The expected result is, that `full` and `zero` should be roughly as fast
# (same range/magnitude). `none` will be much faster and yields the same results as
# `zero`.
print(
    "\n\nThese are the observed execution times (in seconds) for \n"
    f"different combinations of filter-lengths and `case`:\n{results}"
)
