import numpy as np
from collections import deque

# define some buffer to fill, view and consume
class Buffer():

    def __init__(self, maxlen=1000):
        self.timestamps = deque(maxlen=maxlen)
        self.values = deque(maxlen=maxlen)
        self.uncertainties = deque(maxlen=maxlen)
    
    def append(self, time=0.0, value=0.0, uncertainty=0.0):
        self.timestamps.append(time)
        self.values.append(value)
        self.uncertainties.append(uncertainty)
    
    def append_multi(self, timestamps, values, uncertainties):

        for t, v, u in zip(timestamps, values, uncertainties):
            self.append(t, v, u)

    def popleft(self):
        t = self.timestamps.popleft()
        v = self.values.popleft()
        u = self.uncertainties.popleft()
        return t, v, u

    def view_last(self, n):
        r = range(-n,0)

        t = np.array([self.timestamps[i] for i in r])
        v = np.array([self.values[i] for i in r])
        u = np.array([self.uncertainties[i] for i in r])

        return t, v, u

