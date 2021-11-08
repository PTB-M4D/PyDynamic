"""Example how the discrete wavelet transformation can run continuously

This script demonstrates a continuous run of :func:`wave_dec_realtime` by
repetitively calling it.

The script runs infinitely and can be stopped by pressing "Ctrl + c".
"""
import random

import matplotlib.pyplot as plt
import numpy as np
from time_series_buffer import TimeSeriesBuffer as Buffer

import PyDynamic.uncertainty.propagate_DWT as wavelet


def main():
    # basics
    buffer_length = 120
    signal = Buffer(maxlen=buffer_length, return_type="arrays")
    cycle_duration = 0.1  # seconds
    plot_counter = 0
    dwt_counter = 0
    cycle_counter = 0

    # init wavelet stuff
    ld, hd, _, _ = wavelet.filter_design("db2")
    dwt_length = 21

    # init multi level wavelet stuff
    n_levels = 4
    output_multi_level = [n_levels] + [
        level for level in list(range(1, n_levels + 1))[::-1]
    ]  # highest level twice because we store detail + approx
    output_multi_buffer_maxlen = [
        buffer_length // 2 ** level for level in output_multi_level
    ]
    output_multi = [
        Buffer(maxlen=maxlen, return_type="arrays")
        for maxlen in output_multi_buffer_maxlen
    ]  # list of buffer (different lengths to approximately cover the same timespan)
    level_states = None

    # init plot
    plt.ion()
    fig = plt.figure()
    ax = fig.subplots(nrows=1 + len(output_multi), ncols=1, sharex=True)

    # init signal plot
    ax[0].set_ylabel("$x^{(0)}$")
    ax[0].set_ylim([-2, 2])
    (sm,) = ax[0].plot(0, 0, "-k")  # signal
    (su,) = ax[0].plot(0, 0, ":k", linewidth=0.8)  # upper unc
    (sl,) = ax[0].plot(0, 0, ":k", linewidth=0.8)  # lower unc

    # init coefficient plots
    c_lines = []
    for i, (level, _ax) in enumerate(zip(output_multi_level[::-1], ax[1:])):
        if i == len(ax[1:]) - 1:
            _ax.set_ylabel("$a^{{({0})}}$".format(level))
        else:
            _ax.set_ylabel("$d^{{({0})}}$".format(level))
        _ax.set_ylim(auto=True)
        ce = _ax.errorbar(
            0, 0, yerr=0, linewidth=0, elinewidth=2, color="gray", capsize=5
        )
        cm = _ax.scatter([0], [0], c="r", s=10)  # (detail) coeffs
        c_lines.append([cm, ce])
        _ax.set_ylim([-3, 3])

    # simulate infinite stream of data
    while True:
        cycle_counter += 1

        # ti = tm.time()
        ti = cycle_counter * cycle_duration
        ui = 0.15 * (np.sin(2 * ti) + 2)
        xi = np.sin(ti) + np.random.randn() * ui

        signal.add(time=ti, val=xi, val_unc=ui)

        # run DWT every <dwt_length> iterations
        dwt_counter += 1
        if dwt_counter % dwt_length == 0:
            t, _, v, u = signal.show(dwt_length)

            # multi level dwt with uncertainty
            coeffs, Ucoeffs, _, level_states = wavelet.wave_dec_realtime(
                v, u, ld, hd, n=n_levels, level_states=level_states
            )

            # assign correct timestamps to the coefficients
            i0_old = (level_states["counter"] - len(t)) % 2 ** n_levels
            time_indices = np.arange(i0_old, i0_old + len(t))

            # save results to data structure
            for c, u, buffer, level in zip(
                coeffs, Ucoeffs, output_multi, output_multi_level
            ):
                time_indices_level = (time_indices + 1) % 2 ** level == 0
                if (
                    time_indices_level.sum() > 0
                ):  # otherwise error from time-series-buffer
                    buffer.add(time=t[time_indices_level], val=c, val_unc=u)

            dwt_counter = 0

            # set dwt length until next cycle
            dwt_length = random.choice(
                [1, 2, 10, 21]
            )  # could also be constant, e.g. dwt_length = 20
            print(dwt_length)

        # update plot every 5 iterations
        plot_counter += 1
        if (
            plot_counter % 5 == 0 and cycle_counter > buffer_length
        ):  # skip plotting at startup, when buffer is still not fully filled
            # update plot

            # get data to plot
            t_signal, _, v_signal, u_signal = signal.show(-1)

            # update signal lines
            sm.set_xdata(t_signal)
            su.set_xdata(t_signal)
            sl.set_xdata(t_signal)
            sm.set_ydata(v_signal)
            su.set_ydata(v_signal + u_signal)
            sl.set_ydata(v_signal - u_signal)
            ax[0].set_xlim([min(t_signal), max(t_signal)])

            # update dwt lines
            for buffer, _ax, c_line in zip(output_multi[::-1], ax[1:], c_lines):
                if len(buffer) > 0:  # otherwise error from time-series-buffer
                    t_coeff, _, v_coeff, u_coeff = buffer.show(-1)

                    # change the scatter
                    data = np.c_[t_coeff, v_coeff]
                    c_line[0].set_offsets(data)
                    c_line[0].set_facecolor(["r"] * len(t_coeff))

                    # change the error bars
                    c_line[1].remove()
                    c_line[1] = _ax.errorbar(
                        t_coeff,
                        v_coeff,
                        yerr=u_coeff,
                        linewidth=0,
                        elinewidth=1.5,
                        color="gray",
                        capsize=3,
                        zorder=0,
                    )
                    upper_unc = v_coeff + u_coeff
                    lower_unc = v_coeff - u_coeff

                    if v_coeff.size != 0:
                        lim = [np.min(lower_unc), np.max(upper_unc)]
                        _ax.set_ylim(lim)

            # finally update the plot itself
            fig.tight_layout()
            fig.align_ylabels(ax)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # reset plot counter
            plot_counter = 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.\n")
