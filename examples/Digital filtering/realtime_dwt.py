import sys
sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scs
import time as tm

from PyDynamic.misc.testsignals import rect
from PyDynamic.misc.buffer import Buffer
import PyDynamic.uncertainty.propagate_DWT as wavelet


def main():

    # basics
    buffer_length = 200
    signal = Buffer(buffer_length)
    cycle_duration = 0.1  # seconds
    plot_counter = 0
    dwt_counter = 0
    cycle_counter = 0

    # init wavelet stuff
    #output = Buffer(buffer_length // 2)
    #states = None
    ld, hd, lr, hr = wavelet.filter_design("db5")
    dwt_length = 21

    # init multi level wavelet stuff 
    n_levels = 3
    output_multi_level = [n_levels] + [level for level in list(range(1,n_levels+1))[::-1]]    # highest level twice because we store detail + approx
    output_multi_buffer_maxlen = [buffer_length // 2**level for level in output_multi_level]
    output_multi = [Buffer(maxlen=maxlen) for maxlen in output_multi_buffer_maxlen]   # list of buffer (different lengths to approximately cover the same timespan)
    level_states = None

    # init plot
    plt.ion()
    fig = plt.figure()
    ax = fig.subplots(nrows=1+len(output_multi), ncols=1, sharex=True)

    # init signal plot
    ax[0].set_ylim([-2, 2])
    sm, = ax[0].plot(0, 0, "-k")  # signal
    su, = ax[0].plot(0, 0, ":k")  # upper unc
    sl, = ax[0].plot(0, 0, ":k")  # lower unc

    # init coefficient plots
    c_lines = []
    for i, _ax in enumerate(ax[1:]):
        _ax.set_ylim(auto=True)
        cm, = _ax.plot(0, 0, linewidth=0, marker="o", markerfacecolor="r", markeredgecolor="r")  # (detail) coeffs
        cu, = _ax.plot(0, 0, linewidth=0, marker="^", markerfacecolor="r", markeredgecolor="r")  # upper unc
        cl, = _ax.plot(0, 0, linewidth=0, marker="v", markerfacecolor="r", markeredgecolor="r")  # lower unc
        c_lines.append([cm, cu, cl])
        _ax.set_ylim([-3,3])

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

        # run DWT every <dwt_length> iterations
        dwt_counter += 1
        if dwt_counter % dwt_length == 0:
            t, v, u = signal.view_last(dwt_length)

            ## single decomposition with uncertainty
            #c_approx, U_approx, c_detail, U_detail, states = wavelet.dwt(v, u, ld, hd, kind="diag", states=states, realtime=True)
            ## save result to data structure
            #output.append_multi(t[1::2], c_detail, U_detail)
            
            # multi level dwt pre-alpha
            coeffs, Ucoeffs, ol, level_states = wavelet.wave_dec_realtime(v, u, ld, hd, n=n_levels, kind="diag", level_states=level_states)
            
            # assign correct timestamps to the coefficients
            i0_old = (level_states["counter"] - len(t)) % 2**n_levels
            time_indices = np.arange(i0_old, i0_old + len(t))

            # save results to data structure
            for c, u, buffer, level in zip(coeffs, Ucoeffs, output_multi, output_multi_level):
                time_indices_level = (time_indices + 1) % 2**level == 0
                buffer.append_multi(t[time_indices_level], c, u)

            dwt_counter = 0

        # update plot every 5 iterations
        plot_counter += 1
        if plot_counter % 5 == 0:
            # update plot

            ## get data to plot
            t_signal = np.array(signal.timestamps)
            v_signal = np.array(signal.values)
            u_signal = np.array(signal.uncertainties)

            ## update signal lines
            sm.set_xdata(t_signal)
            su.set_xdata(t_signal)
            sl.set_xdata(t_signal)
            sm.set_ydata(v_signal)
            su.set_ydata(v_signal+u_signal)
            sl.set_ydata(v_signal-u_signal)
            ax[0].set_xlim([min(signal.timestamps), max(signal.timestamps)])

            # update dwt lines
            for buffer, _ax, c_line in zip(output_multi[::-1], ax[1:], c_lines[::-1]):
                t_coeff = np.array(buffer.timestamps)
                v_coeff = np.array(buffer.values)
                u_coeff = np.array(buffer.uncertainties)

                c_line[0].set_xdata(t_coeff)
                c_line[1].set_xdata(t_coeff)
                c_line[2].set_xdata(t_coeff)
                c_line[0].set_ydata(v_coeff)
                c_line[1].set_ydata(v_coeff+u_coeff)
                c_line[2].set_ydata(v_coeff-u_coeff)

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