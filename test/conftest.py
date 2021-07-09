import os

from hypothesis import HealthCheck, settings

# This will check, if the testrun is executed in the ci environment and if so,
# disables the 'too_slow' health check. See
# https://hypothesis.readthedocs.io/en/latest/healthchecks.html#hypothesis.HealthCheck
# for some details.
settings.register_profile("ci", suppress_health_check=(HealthCheck.too_slow,))
if "CIRCLECI" in os.environ:
    settings.load_profile("ci")
