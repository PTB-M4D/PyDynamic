"""This module implements the signals class and its derivatives

Signals are dynamic quantities with associated uncertainties, quantity and time
units. A signal has to be defined together with a time axis.

.. note:: This module is work in progress!
"""

__all__ = ["Signal"]

from math import isclose
from typing import Optional, Union

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
    """Signal class which represents a common signal in digital signal processing

    Parameters
    ----------
    time: np.ndarray
        the time axis as :class:`np.ndarray <numpy.ndarray>` of floats,
        number of elements must coincide with number of values
    values: np.ndarray
        signal values' magnitudes, number of elements must coincide with number of
        elements in time
    Ts: float, optional
        the sampling interval length, i.e. the difference between each two time stamps,
        defaults to the reciprocal of the sampling frequency if provided and the mean of
        all unique interval lengths otherwise
    Fs: float, optional
        the sampling frequency, defaults to the reciprocal of the sampling interval
        length
    uncertainty: float or np.ndarray, optional
        the uncertainties associated with the signal values, depending on the type and
        shape the following should be provided:

        - float: constant standard uncertainty for all values
        - 1D-array: element-wise standard uncertainties
        - 2D-array: covariance matrix
    """

    _unit_time: str
    _unit_values: str
    _name: str
    _uncertainty: np.ndarray
    _standard_uncertainties: np.ndarray
    _Ts: float
    _Fs: float
    _time: np.ndarray
    _values: np.ndarray

    def __init__(
        self,
        time: np.ndarray,
        values: np.ndarray,
        Ts: Optional[float] = None,
        Fs: Optional[float] = None,
        uncertainty: Optional[Union[float, np.ndarray]] = None,
    ):
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
        self._time = time
        self._values = values
        if Ts is not None and Fs is not None and not isclose(Fs, 1 / Ts):
            raise ValueError(
                "Signal: Sampling interval and sampling frequency are assumed to "
                "be approximately multiplicative inverse to each other, but "
                f"Fs={Fs} and Ts={Ts}. Please adjust either one of them."
            )
        if Ts is None and Fs is None:
            self._Ts = np.unique(np.diff(self.time)).mean()
            self._Fs = 1 / self._Ts
        elif isinstance(Fs, float):
            self._Ts = 1 / Fs
            self._Fs = Fs
        else:
            self._Fs = 1 / Ts
            self._Ts = Ts
        self.uncertainty = uncertainty
        self.set_labels()

    def set_labels(self, unit_time="s", unit_values="a.u.", name_values="signal"):
        self._unit_time = unit_time
        self._unit_values = unit_values
        self._name = name_values

    def plot(self, fignr=1, figsize=(10, 8)):
        figure(fignr, figsize=figsize)
        plot(self.time, self.values, label=self.name)
        fill_between(
            self.time,
            self.values - self.standard_uncertainties,
            self.values + self.standard_uncertainties,
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
            self.standard_uncertainties,
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
        r"""Apply digital filter (b, a) to the signal values

        Apply digital filter (b, a) to the signal values and propagate the
        uncertainty associated with the signal. Time vector is assumed to be
        equidistant, as well as corresponding values should represent evenly spaced
        signal magnitudes.

        Parameters
        ----------
        b : np.ndarray
            filter numerator coefficients
        a : np.ndarray, optional
            filter denominator coefficients, defaults to :math:`a=(1)` for FIR-type
            filter
        filter_uncertainty : np.ndarray, optional
            For IIR-type filter provide covariance matrix :math:`U_{\theta}`
            associated with filter coefficient vector :math:`\theta=(a_1,\ldots,a_{N_a},
            b_0,\ldots,b_{N_b})^T`. For FIR-type filter provide one of the following:

            - 1D-array: coefficient-wise standard uncertainties of filter
            - 2D-array: covariance matrix associated with theta

            if the filter is fully certain, use `filter_uncertainty = None` (default)
            to make use of more efficient calculations.

        MonteCarloRuns : int, optional
            number of Monte Carlo runs, defaults to 10.000, only considered for
            IIR-type filters. Otherwise :func:`FIRuncFilter
            <PyDynamic.uncertainty.propagate_filter.FIRuncFilter>` is applied directly
        """

        if self._is_fir_type_filter(a):
            self._values, self.uncertainty = FIRuncFilter(
                self.values, self.uncertainty, b, Utheta=filter_uncertainty, kind="diag"
            )
        else:
            self._values, self.uncertainty = MC(
                self.values,
                self.uncertainty,
                b,
                a,
                filter_uncertainty,
                runs=MonteCarloRuns,
            )

    @staticmethod
    def _is_fir_type_filter(a: np.ndarray) -> bool:
        return len(a) == 1 and a[0] == 1

    @property
    def Ts(self) -> float:
        """Sampling interval, i.e. (averaged) difference between each two time stamps"""
        return self._Ts

    @property
    def Fs(self) -> float:
        """Sampling frequency, i.e. the sampling interval :attr:`Ts`' reciprocal"""
        return self._Fs

    @property
    def unit_time(self) -> str:
        """Unit of the :attr:`time` vector"""
        return self._unit_time

    @property
    def unit_values(self) -> str:
        """Unit of the :attr:`values` vector"""
        return self._unit_values

    @property
    def name(self) -> str:
        """Signal name"""
        return self._name

    @property
    def standard_uncertainties(self) -> np.ndarray:
        """Element-wise standard uncertainties associated to :attr:`values`"""
        return self._standard_uncertainties

    @property
    def uncertainty(self) -> np.ndarray:
        """Uncertainties associated with the signal :attr:`values`

        Depending on the uncertainties provided during initialization, one of following
        will be provided:

        - 1D-array: element-wise standard uncertainties
        - 2D-array: covariance matrix
        """
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value: Union[float, np.ndarray]):
        if isinstance(value, float):
            self._uncertainty = np.full_like(self.values, value)
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
                    f"{uncertainties_array.shape} and time is of length "
                    f"{len(self.time)}. Please adjust either one of them."
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
                self._standard_uncertainties = np.sqrt(np.diag(uncertainties_array))
        else:
            self._uncertainty = np.zeros_like(self.values)
            self._standard_uncertainties = self._uncertainty

    @property
    def time(self) -> np.ndarray:
        """Signal's time axis"""
        return self._time

    @property
    def values(self) -> np.ndarray:
        """Signal values' magnitudes"""
        return self._values
