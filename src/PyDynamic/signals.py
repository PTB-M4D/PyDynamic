"""This module implements the signals class and its derivatives

Signals are dynamic quantities with associated uncertainties. A signal has to be defined
together with a time axis.

.. note:: This module is experimental!
"""

__all__ = ["Signal"]

from math import isclose
from typing import Optional

import numpy as np
from matplotlib.pyplot import figure, fill_between, legend, plot, xlabel, ylabel

from .misc.tools import (
    is_2d_matrix,
    is_2d_square_matrix,
    is_vector,
    number_of_rows_equals_vector_dim,
)
from .uncertainty.propagate_filter import FIRuncFilter
from .uncertainty.propagate_MonteCarlo import MC


class Signal:
    """The base class which defines the interfaces and default behaviour."""

    unit_time = ""
    unit_values = ""
    name = ""

    def __init__(self, time, values, Ts=None, Fs=None, uncertainty=None):
        if len(values.shape) > 1:
            raise NotImplementedError(
                "Signal: Multivariate signals are not implemented yet."
            )
        if len(time) != len(values):
            raise ValueError(
                "Signal: Number of elements of the provided time and signal vectors "
                f"are expected to match, but time is of length {len(time)} and values "
                f"is of length {len(values)}. Please adjust either one of them."
            )
        self.time = time
        self.values = values
        # set sampling interval and frequency
        if (Ts is None) and (Fs is None):
            self.Ts = np.unique(np.diff(self.time)).mean()
            self.Fs = 1 / self.Ts
        elif isinstance(Ts, float):
            if Fs is None:
                self.Fs = 1 / Ts
            elif not isclose(Fs, 1 / Ts):
                raise ValueError(
                    "Signal: Sampling interval and sampling frequency are assumed to "
                    "be approximately multiplicative inverse to each other, but "
                    f"Fs={Fs} and Ts={Ts}. Please adjust either one of them."
                )
            self.Ts = Ts
        self.uncertainty = uncertainty
        self.set_labels()

    def set_labels(self, unit_time="s", unit_values="a.u.", name_values="signal"):
        self.unit_time = unit_time
        self.unit_values = unit_values
        self.name = name_values

    def plot(self, fignr=1, figsize=(10, 8)):
        figure(fignr, figsize=figsize)
        plot(self.time, self.values, label=self.name)
        fill_between(
            self.time,
            self.values - self._standard_uncertainties,
            self.values + self._standard_uncertainties,
            color="gray",
            alpha=0.2,
        )
        xlabel("time / %s" % self.unit_time)
        ylabel("%s / %s" % (self.name, self.unit_values))
        legend(loc="best")

    def plot_uncertainty(self, fignr=2, **kwargs):
        figure(fignr, **kwargs)
        plot(
            self.time,
            self._standard_uncertainties,
            label="uncertainty associated with %s" % self.name,
        )
        xlabel("time / %s" % self.unit_time)
        ylabel("uncertainty / %s" % self.unit_values)
        legend(loc="best")

    def apply_filter(
        self,
        b: np.ndarray,
        a: Optional[np.ndarray] = np.ones(1),
        filter_uncertainty: Optional[np.ndarray] = None,
        MonteCarloRuns: Optional[int] = 10000,
    ):
        """Apply digital filter (b, a) to the signal values

        Apply digital filter (b, a) to the signal values and propagate the
        uncertainty associated with the signal.

        Parameters
        ----------
        b : np.ndarray
            filter numerator coefficients
        a : np.ndarray, optional
            filter denominator coefficients, defaults to :math:`a=(1)` for FIR-type
            filter
        filter_uncertainty : np.ndarray, optional
            For IIR-type filter provide covariance matrix associated with filter
            coefficients. For FIR-type filter provide one of the following

            - 1D-array: coefficient-wise standard uncertainties of filter
            - 2D-array: covariance matrix associated with theta

            if the filter is fully certain, use `Utheta = None` (default) to make use
            of more efficient calculations.

        MonteCarloRuns : int, optional
            number of Monte Carlo runs, defaults to 10.000
        """

        if self._is_fir_type_filter(a):
            self.values, self.uncertainty = FIRuncFilter(
                self.values, self.uncertainty, b, Utheta=filter_uncertainty, kind="diag"
            )
        else:  # IIR-type filter
            if not isinstance(MonteCarloRuns, int):
                MonteCarloRuns = 10000
            self.values, self.uncertainty = MC(
                self.values,
                self.uncertainty,
                b,
                a,
                filter_uncertainty,
                runs=MonteCarloRuns,
            )

    @staticmethod
    def _is_fir_type_filter(a):
        return len(a) == 1 and a[0] == 1

    @property
    def uncertainty(self):
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        if isinstance(value, float):
            self._uncertainty = np.ones_like(self.values) * value
            self._standard_uncertainties = self._uncertainty
        elif isinstance(value, np.ndarray):
            uncertainties_array = value.squeeze()
            if not number_of_rows_equals_vector_dim(
                matrix=uncertainties_array, vector=self.time
            ):
                raise ValueError(
                    "Signal: if uncertainties are provided as np.ndarray "
                    f"they are expected to match the number of elements of the "
                    f"provided time vector, but uncertainties are of shape "
                    f"{uncertainties_array.shape} and time is of length {len(self.time)}. "
                    f"Please adjust either one of them."
                )
            if is_2d_matrix(uncertainties_array) and not is_2d_square_matrix(
                uncertainties_array
            ):
                raise ValueError(
                    "Signal: if uncertainties are provided as 2-dimensional np.ndarray "
                    f"they are expected to resemble a square matrix, but uncertainties "
                    f"are of shape {uncertainties_array.shape}. Please "
                    f"adjust them."
                )
            self._uncertainty = uncertainties_array
            if is_vector(uncertainties_array):
                self._standard_uncertainties = uncertainties_array
            else:
                self._standard_uncertainties = np.diag(uncertainties_array)
        else:
            self._uncertainty = np.zeros_like(self.values)
            self._standard_uncertainties = self._uncertainty
