r"""
The :mod:`PyDynamic.misc.buffer` module is a collection of methods which are
related to buffering.

This module contains the following functions:

* :class:`TimeSeriesBuffer`: Simple FIFO buffer class with uncertainty for time-series applications
"""

from collections import deque

import numpy as np

__all__ = [
    "TimeSeriesBuffer",
]


class TimeSeriesBuffer:
    """
    Define some buffer to fill, view and consume.
    """

    def __init__(self, maxlen=1000):
        """Initialize a FIFO buffer.
        
        Parameters
        ----------
            maxlen: int (default: 1000)
                maximum length of the buffer, directly handed over to deque
        
        """
        self.timestamps = deque(maxlen=maxlen)
        self.values = deque(maxlen=maxlen)
        self.uncertainties = deque(maxlen=maxlen)

    def append(self, time=0.0, value=0.0, uncertainty=0.0):
        """Append a new entry datapoint to the buffer. The data point
        consists of the triple (time, value, uncertainty).
        
        Parameters
        ----------
            time: float (default: 0.0)
                timestamp of the triple
            value: float (default: 0.0)
                value of the datapoint
            uncertainty: float (default: 0.0)
                uncertainty of value
        
        """
        self.timestamps.append(time)
        self.values.append(value)
        self.uncertainties.append(uncertainty)

    def append_multi(self, timestamps, values, uncertainties):
        """Append multiple datapoints to buffer. 
        
        Parameters
        ----------
            timestamps: list or np.ndarray of float
                Iterable list/array of length `N` with timestamps.
            values: list or np.ndarray of float
                Iterable list/array of length `N` with values.
            uncertainties: list or np.ndarray of float
                Iterable list/array of length `N` with uncertainties.
        
        """
        for t, v, u in zip(timestamps, values, uncertainties):
            self.append(t, v, u)

    def popleft(self):
        """Pop the oldest datapoint from the buffer and return it.
        
        Returns
        -------
            t: float
                timestamps of popped datapoint
            v: float
                values of popped datapoint
            u: float
                uncertainties of popped datapoint
        """
        t = self.timestamps.popleft()
        v = self.values.popleft()
        u = self.uncertainties.popleft()
        return t, v, u

    def view_last(self, n=1):
        """View the latest `n` additions to the buffer. Returns the same format that
        :py:func:`append_multi` accepts.
        
        Parameters
        ----------
            n: int (default: 1)
                How many datapoints to return.
        
        Returns
        -------
            t: np.ndarray
                array of length `n` with timestamps.
            v: np.ndarray
                array of length `n` with values.
            u: np.ndarray
                array of length `n` with uncertainties.
        """
        r = range(-n, 0)

        t = np.array([self.timestamps[i] for i in r])
        v = np.array([self.values[i] for i in r])
        u = np.array([self.uncertainties[i] for i in r])

        return t, v, u
