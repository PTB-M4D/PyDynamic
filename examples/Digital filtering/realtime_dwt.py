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

        t = [self.timestamps[i] for i in r]
        v = [self.values[i] for i in r]
        u = [self.uncertainties[i] for i in r]

        return t, v, u


def main():

    # basics
    signal = Buffer(200)
    cycle_duration = 0.1  # seconds
    plot_counter = 0
    dwt_counter = 0
    cycle_counter = 0

    # init wavelet stuff
    output = Buffer(100)
    ld, hd, lr, hr = wavelet.filter_design("db5")
    states = None

    # init plot
    plt.ion()
    fig = plt.figure()
    ax = fig.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].set_ylim([-2, 2])
    ax[1].set_ylim([-2, 2])

    sm, = ax[0].plot(0, 0, "-k")  # signal
    su, = ax[0].plot(0, 0, ":k")  # upper unc
    sl, = ax[0].plot(0, 0, ":k")  # lower unc

    cm, = ax[0].plot(0, 0, "-r")  # approx coeffs
    cu, = ax[0].plot(0, 0, ":r")  # approx coeffs
    cl, = ax[0].plot(0, 0, ":r")  # approx coeffs

    # simulate infinite stream of data
    while True:
        cycle_counter += 1
        # log when cycle started
        #cycle_start = tm.time()

        #ti = tm.time()
        ti = cycle_counter * cycle_duration
        ui = 0.05 * (np.sin(2*ti) + 2)
        xi = np.sin(ti) + np.random.randn() * ui

        signal.append(ti, xi, ui)

        # run DWT every 5 iterations
        dwt_counter += 1
        if dwt_counter % 20 == 0:
            t, v, u = signal.view_last(20)

            # single decomposition with uncertainty
            c_approx, U_approx, c_detail, U_detail, states = wavelet.dwt(v, u, ld, hd, kind="diag", states=states, realtime=True)

            # save result to data structure
            output.append_multi(t[1::2], c_approx, U_approx)

            dwt_counter = 0


        # update plot every 10 iterations
        plot_counter += 1
        if plot_counter % 5 == 0:
            # update plot

            ## get data to plot
            t_signal = np.array(signal.timestamps)
            v_signal = np.array(signal.values)
            u_signal = np.array(signal.uncertainties)
            t_output = np.array(output.timestamps)
            v_output = np.array(output.values)
            u_output = np.array(output.uncertainties)

            ## update signal lines
            sm.set_xdata(t_signal)
            su.set_xdata(t_signal)
            sl.set_xdata(t_signal)
            sm.set_ydata(v_signal)
            su.set_ydata(v_signal+u_signal)
            sl.set_ydata(v_signal-u_signal)
            ax[0].set_xlim([min(signal.timestamps), max(signal.timestamps)])

            # update dwt lines
            cm.set_xdata(t_output)
            cu.set_xdata(t_output)
            cl.set_xdata(t_output)
            cm.set_ydata(v_output)
            cu.set_ydata(v_output+u_output)
            cl.set_ydata(v_output-u_output)
            ax[1].set_xlim([min(signal.timestamps), max(signal.timestamps)])

            fig.canvas.draw()
            fig.canvas.flush_events()

            # reset plot counter
            plot_counter = 0

        # wait the rest until
        #real_duration = tm.time() - cycle_start
        #sleep_duration = cycle_duration - real_duration
        #if sleep_duration >= 0.0:
        #    tm.sleep(sleep_duration)
        #else:
        #    print("Warning, cycle took longer than given length. (real: {0:.3f}s, target: {1:.3f})".format(real_duration, cycle_duration))
        tm.sleep(cycle_duration)





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.\n")