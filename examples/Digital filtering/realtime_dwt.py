import sys
sys.path.append(".")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import itertools
import numpy as np
import scipy.signal as scs
import time as tm
import random

from PyDynamic.misc.testsignals import rect
from PyDynamic.misc.buffer import Buffer
import PyDynamic.uncertainty.propagate_DWT as wavelet


def main():

    # basics
    buffer_length = 120
    signal = Buffer(buffer_length)
    cycle_duration = 0.1  # seconds
    plot_counter = 0
    dwt_counter = 0
    cycle_counter = 0

    # init wavelet stuff
    #output = Buffer(buffer_length // 2)
    #states = None
    ld, hd, lr, hr = wavelet.filter_design("db2")
    dwt_length = 21

    # init multi level wavelet stuff 
    n_levels = 4
    output_multi_level = [n_levels] + [level for level in list(range(1,n_levels+1))[::-1]]    # highest level twice because we store detail + approx
    output_multi_buffer_maxlen = [buffer_length // 2**level for level in output_multi_level]
    output_multi = [Buffer(maxlen=maxlen) for maxlen in output_multi_buffer_maxlen]   # list of buffer (different lengths to approximately cover the same timespan)
    level_states = None

    # init plot
    plt.ion()
    fig = plt.figure()
    ax = fig.subplots(nrows=1+len(output_multi), ncols=1, sharex=True)

    # init signal plot
    ax[0].set_ylabel("x")
    ax[0].set_ylim([-2, 2])
    sm, = ax[0].plot(0, 0, "-k")  # signal
    su, = ax[0].plot(0, 0, ":k")  # upper unc
    sl, = ax[0].plot(0, 0, ":k")  # lower unc

    # init coefficient plots
    c_lines = []
    for i, (level, _ax) in enumerate(zip(output_multi_level[::-1], ax[1:])):
        if i == len(ax[1:]) - 1:
            _ax.set_ylabel("A^({0})".format(level))
        else:
            _ax.set_ylabel("D^({0})".format(level))
        _ax.set_ylim(auto=True)
        cm = _ax.scatter([0], [0], c='r')  # (detail) coeffs
        cu, = _ax.plot(0, 0, ":r", linewidth=0.5)  # upper unc
        cl, = _ax.plot(0, 0, ":r", linewidth=0.5)  # lower unc
        c_lines.append([cm, cu, cl])
        _ax.set_ylim([-3,3])

    # simulate infinite stream of data
    while True:
        cycle_counter += 1
        # log when cycle started
        #cycle_start = tm.time()

        #ti = tm.time()
        ti = cycle_counter * cycle_duration
        ui = 0.15 * (np.sin(2*ti) + 2)
        xi = np.sin(ti) + np.random.randn() * ui

        signal.append(ti, xi, ui)

        # run DWT every <dwt_length> iterations
        dwt_counter += 1
        if dwt_counter % dwt_length == 0:
            t, v, u = signal.view_last(dwt_length)
            
            # multi level dwt with uncertainty
            coeffs, Ucoeffs, ol, level_states = wavelet.wave_dec_realtime(v, u, ld, hd, n=n_levels, kind="diag", level_states=level_states)
            
            # assign correct timestamps to the coefficients
            i0_old = (level_states["counter"] - len(t)) % 2**n_levels
            time_indices = np.arange(i0_old, i0_old + len(t))

            # save results to data structure
            for c, u, buffer, level in zip(coeffs, Ucoeffs, output_multi, output_multi_level):
                time_indices_level = (time_indices + 1) % 2**level == 0
                buffer.append_multi(t[time_indices_level], c, u)

            dwt_counter = 0

            # change dwt length until next cycle, TESTING!!!
            dwt_length = random.choice([1, 2, 10, 21])
            print(dwt_length)

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
            for buffer, _ax, c_line in zip(output_multi[::-1], ax[1:], c_lines):
                t_coeff = np.array(buffer.timestamps)
                v_coeff = np.array(buffer.values)
                u_coeff = np.array(buffer.uncertainties)

                # change the scatter
                data = np.c_[t_coeff, v_coeff]
                c_line[0].set_offsets(data)
                c_line[0].set_facecolor(["r"]*len(t_coeff))

                # change upper unc line
                upper_unc = v_coeff+u_coeff
                c_line[1].set_xdata(t_coeff)
                c_line[1].set_ydata(upper_unc)
                
                # change lower unc line
                lower_unc = v_coeff-u_coeff
                c_line[2].set_xdata(t_coeff)
                c_line[2].set_ydata(lower_unc)

                if v_coeff.size != 0:
                    lim = [np.min(lower_unc), np.max(upper_unc)]
                    _ax.set_ylim(lim)
            
            # highlight the largest coefficients
            coeffs_all = [np.array(buffer.values) for i_buffer, buffer in enumerate(output_multi)]
            coeffs_all_indices = [[(i_buffer, i_coeff) for i_coeff, coeff in enumerate(buffer.values)] for i_buffer, buffer in enumerate(output_multi)]
            coeffs_joined = list(itertools.chain(*coeffs_all))                      # make 1D list of all coeff-values
            coeffs_joined_indices = list(itertools.chain(*coeffs_all_indices))      # make 1D list of (i_buffer, i_coeff)
            coeff_highest = np.argpartition(-np.abs(coeffs_joined), kth=range(10))[:10]  # get 10 biggest (absolut) values
            coeff_highest_indicies = [coeffs_joined_indices[i] for i in coeff_highest]
            
            for coeff_index in coeff_highest_indicies:
                _ax = ax[1::][::-1][coeff_index[0]]
                c_line = c_lines[::-1][coeff_index[0]]
                
                tmp_colors = c_line[0].get_facecolor()
                tmp_colors = ["k" if i == coeff_index[1] else c for i, c in enumerate(tmp_colors)]
                c_line[0].set_facecolor(tmp_colors)

            # finally update the plot itself
            fig.tight_layout()
            fig.align_ylabels(ax)
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
        tm.sleep(cycle_duration/2)





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.\n")