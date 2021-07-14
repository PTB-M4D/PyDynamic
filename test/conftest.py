import os

import numpy as np
from hypothesis import HealthCheck, settings

# This will check, if the testrun is executed in the ci environment and if so,
# disables the 'too_slow' health check. See
# https://hypothesis.readthedocs.io/en/latest/healthchecks.html#hypothesis.HealthCheck
# for some details.
settings.register_profile(name="ci", suppress_health_check=(HealthCheck.too_slow,))
if "CIRCLECI" in os.environ:
    settings.load_profile("ci")


def random_covariance_matrix(length):
    """construct a valid (but random) covariance matrix with good condition number"""

    # because np.cov estimates the mean from data, the returned covariance matrix
    # has one eigenvalue close to numerical zero (rank n-1).
    # This leads to a singular matrix, which is badly suited to be used as valid
    # covariance matrix. To circumvent this:

    # draw random (n+1, n+1) matrix
    cov = np.cov(np.random.random((length + 1, length + 1)))

    # calculate SVD
    u, s, vh = np.linalg.svd(cov, full_matrices=False, hermitian=True)

    # reassemble a covariance of size (n, n) by discarding the smallest singular value
    cov_adjusted = (u[:-1, :-1] * s[:-1]) @ vh[:-1, :-1]

    return cov_adjusted
