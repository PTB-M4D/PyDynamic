import sys
sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scs
import time as tm
from collections import deque

from PyDynamic.misc.testsignals import rect
import PyDynamic.uncertainty.propagate_DWT as wavelet

# define some buffer to fill and consume
class Buffer():

    def __init__(self, maxlen=1000):
        self.timestamps = deque(maxlen=maxlen)
        self.values = deque(maxlen=maxlen)
        self.uncertainties = deque(maxlen=maxlen)
    
    def append(self, time=0.0, value=0.0, uncertainty=0.0):
        self.timestamps.append(time)
        self.values.append(value)
        self.uncertainties.append(uncertainty)

    def popleft(self):
        t = self.timestamps.popleft()
        v = self.values.popleft()
        u = self.uncertainties.popleft()

        return t, v, u


def main():

    signal = Buffer(200)
    cycle_period = 0.1  # seconds
    plot_counter = 0

    # init plot
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([-2, 2])

    hm, = ax.plot(0, 0, "-k")  # mean
    hu, = ax.plot(0, 0, ":k")  # upper unc
    hl, = ax.plot(0, 0, ":k")  # lower unc

    # simulate infinite stream of data
    while True:
        # log when cycle started
        cycle_start = tm.time()

        ti = tm.time()
        ui = 0.05 * (np.sin(2*ti) + 2)
        xi = np.sin(ti) + np.random.randn() * ui

        signal.append(ti, xi, ui)

        # update plot every 5 iterations
        plot_counter += 1
        if plot_counter % 5 == 0:
            # update plot
            t = np.array(signal.timestamps)
            v = np.array(signal.values)
            u = np.array(signal.uncertainties)

            # update lines
            hm.set_xdata(t)
            hu.set_xdata(t)
            hl.set_xdata(t)

            hm.set_ydata(v)
            hu.set_ydata(v+u)
            hl.set_ydata(v-u)

            ax.set_xlim([min(signal.timestamps), max(signal.timestamps)])

            fig.canvas.draw()
            fig.canvas.flush_events()

            # reset plot counter
            plot_counter = 0

        # wait the rest until
        duration = tm.time() - cycle_start
        sleep_duration = cycle_period - duration
        if sleep_duration >= 0.0:
            tm.sleep(sleep_duration)
        else:
            print("Warning, cycle took longer than given length. (is: {0:.3f}s, should: {1:.3f})".format(duration, cycle_period))





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.\n")