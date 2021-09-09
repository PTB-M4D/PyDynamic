import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from hypothesis import HealthCheck, settings, strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite, SearchStrategy

# This will check, if the testrun is executed in the ci environment and if so,
# disables the 'too_slow' health check. See
# https://hypothesis.readthedocs.io/en/latest/healthchecks.html#hypothesis.HealthCheck
# for some details.
settings.register_profile(
    name="ci", suppress_health_check=(HealthCheck.too_slow,), deadline=None
)
if "CIRCLECI" in os.environ:
    settings.load_profile("ci")


@dataclass
class VectorAndCompatibleMatrix:
    vector: np.ndarray
    matrix: np.ndarray


def random_float_square_matrix_strategy(
    number_of_rows: int,
) -> SearchStrategy:
    return hnp.arrays(dtype=float, shape=(number_of_rows, number_of_rows))


@composite
def random_float_square_matrix(
    draw: Callable, number_of_rows: Optional[int] = None
) -> np.ndarray:
    number_of_rows_and_columns = (
        number_of_rows
        if number_of_rows is not None
        else draw(hst.integers(min_value=1, max_value=20))
    )
    return draw(
        random_float_square_matrix_strategy(number_of_rows=number_of_rows_and_columns)
    )


@composite
def random_float_vector(draw: Callable, length: Optional[int] = None) -> np.ndarray:
    number_of_elements = (
        length if length is not None else draw(hst.integers(min_value=1, max_value=20))
    )
    return draw(hnp.arrays(dtype=float, shape=number_of_elements))


def random_not_negative_float_strategy() -> SearchStrategy:
    return hst.floats(min_value=0)


@composite
def random_not_negative_float(draw: Callable) -> float:
    return draw(random_not_negative_float_strategy)
