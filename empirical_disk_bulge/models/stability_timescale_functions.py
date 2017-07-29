"""
"""
import numpy as np


def sigmoid(x, x0, y0, height_diff, k):
    """
    """
    ymin = y0 - height_diff/2.
    return ymin + height_diff/(1 + np.exp(-k*(x-x0)))


def stability_timescale_vs_dvmax(dvmax, tau_quenched, tau_star_forming, quenched_fraction, k=10):
    height_diff = tau_star_forming - tau_quenched
    y0 = tau_quenched + height_diff/2.
    x0 = 1. - quenched_fraction
    return sigmoid(dvmax, x0, y0, height_diff, k)


def stability_timescale_vs_vmax(vmax, tau_low_vmax, tau_high_vmax, vmax_char):
    x0 = np.log(vmax_char)
    height_diff = tau_high_vmax - tau_low_vmax
    y0 = tau_low_vmax + height_diff/2.
    return sigmoid(np.log(vmax), x0, y0, height_diff, k=x0)


def quenched_fraction_vs_vmax(vmax, frac_q_low_vmax, frac_q_high_vmax, low_vmax=10, high_vmax=1000):
    log_low_vmax = np.log(low_vmax)
    log_high_vmax = np.log(high_vmax)
    return np.interp(np.log(vmax), (log_low_vmax, log_high_vmax), (frac_q_low_vmax, frac_q_high_vmax))


def stability_timescale(vmax, dvmax, *params):
    """
    """
    vmax_char = params[0]
    tau_low_vmax_q, tau_high_vmax_q = params[1:3]
    tau_q = stability_timescale_vs_vmax(vmax, tau_low_vmax_q, tau_high_vmax_q, vmax_char)

    tau_low_vmax_sf, tau_high_vmax_sf = params[3:5]
    tau_sf = stability_timescale_vs_vmax(vmax, tau_low_vmax_sf, tau_high_vmax_sf, vmax_char)

    frac_q_low_vmax, frac_q_high_vmax = params[5:7]
    fq = quenched_fraction_vs_vmax(vmax, frac_q_low_vmax, frac_q_high_vmax)

    return stability_timescale_vs_dvmax(dvmax, tau_q, tau_sf, fq)
