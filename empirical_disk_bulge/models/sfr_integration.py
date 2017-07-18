"""
"""
import os
import numpy as np


def _cosmic_age_array_midpoints(arr):
    midpoints = 0.5*(arr[:-1] + arr[1:])
    return np.insert(midpoints, 0, midpoints[0])


def in_situ_stellar_mass(sfr_history, cosmic_age_array, t, kernel=None):
    """ Calculate the total stellar mass at time t
    by integrating the input in-situ SFR history,
    accounting for mass-loss due to passive evolution.
    Parameters
    ----------
    sfr_history : ndarray
        Numpy array of shape (num_gals, num_time_steps) storing the
        star-formation rate in units of Msun/yr.

    t : float
        Time at which to calculate the total stellar mass in units of Gyr

    kernel : ndarray, optional
        Numpy array of any shape that can be broadcast against `sfr_history`,
        e.g., a boolean mask or the result of some weighting function

    Returns
    -------
    stellar_mass : ndarray
        Numpy array of shape (num_gals, )
    """
    dt, frac_remaining = _stellar_mass_integrand_factors(t, cosmic_age_array)
    integrand = sfr_history[:, :len(dt)] * dt * frac_remaining
    if kernel is not None:
        integrand = integrand * kernel
    return np.sum(integrand, axis=1) * 1e9


def stellar_mass_interpolation(sm_history, t, cosmic_age_array):
    """ Interpolate between adjacent snapshots to determine the stellar mass.
    """
    msg = "Input ``t`` = {0:.2f} cannot exceed final snapshot = {1:.2f}"
    assert t <= cosmic_age_array[-1], msg.format(t, cosmic_age_array[-1])

    idx_upper_snap = np.searchsorted(cosmic_age_array, t)

    if cosmic_age_array[idx_upper_snap] == t:
        return sm_history[:, idx_upper_snap]
    else:
        t1, t2 = 1e9*cosmic_age_array[idx_upper_snap-1], 1e9*cosmic_age_array[idx_upper_snap]
        sm1 = sm_history[:, idx_upper_snap-1]
        sm2 = sm_history[:, idx_upper_snap]

        slope = (sm2 - sm1)/(t2 - t1)
        dt = 1e9*t - t1
        sm = sm1 + slope*dt
        return sm


def remaining_mass_fraction(t, t_form):
    """
    Fraction of mass lost during time interval dt due to passive evolution.

    Parameters
    ----------
    t : float
        Age of universe at the time for which the remaining mass fraction
        is to be calculated, in units of Gyr.

    t_form : ndarray
        Numpy array of shape (num_time_steps, ) storing
        a discrete set of times at which stars were formed, in units of Gyr.
        Values of t_form may not exceed t.

    Returns
    --------
    mass_fraction : array
        Length N array of the fraction of remaining mass.
    """
    t_form = np.atleast_1d(t_form).astype('f4')
    mask = t_form < t
    result = np.ones_like(t_form)
    result[mask] = 1. - 0.05*np.log(1 + 1000.*(t - t_form[mask])/1.4)
    return result


def in_situ_fraction(sfr_history, sm_mp_history, redshift, cosmic_age_array, kernel=None):
    a = 1./(1. + redshift)
    t = np.interp(a, bolplanck_scale_factors, cosmic_age_array)

    in_situ_sm = in_situ_stellar_mass(sfr_history, t, kernel=kernel)
    total_sm = stellar_mass_interpolation(sm_mp_history, t)
    result = np.zeros_like(total_sm) + np.nan
    mask = total_sm > 0.
    result[mask] = in_situ_sm[mask]/total_sm[mask]
    return result


def _largest_midpoint_index(t, cosmic_age_midpoints):
    """
    """
    assert t > cosmic_age_midpoints[0], "Input time cannot precede first simulation snapshot"
    return np.searchsorted(cosmic_age_midpoints, t)


def _stellar_mass_integrand_factors(t, cosmic_age_array):

    cosmic_age_midpoints = _cosmic_age_array_midpoints(cosmic_age_array)

    idx_largest_midpoint = _largest_midpoint_index(t, cosmic_age_midpoints)

    bin_edges = cosmic_age_midpoints[:idx_largest_midpoint]
    if t > bin_edges[-1]:
        bin_edges = np.append(bin_edges, t)
    dt = np.diff(bin_edges)

    times = cosmic_age_array[:len(dt)]

    frac_remaining = remaining_mass_fraction(t, times)

    return dt, frac_remaining


dirname = "/Users/aphearin/Dropbox/UniverseMachine/data/histories"
basename = "small_sfh_catalog_1.002310.txt"
fname = os.path.join(dirname, basename)
with open(fname, 'r') as f:
    line = next(f)
    line = next(f)
    line = next(f)
    line = next(f)
bolplanck_scale_factors = np.array(list(float(s) for s in line.strip().split()[2:]))
bolplanck_redshifts = 1./bolplanck_scale_factors - 1.


def _index_of_nearest_larger_redshift(z, cosmic_redshift_array):
    return len(cosmic_redshift_array) - np.searchsorted(cosmic_redshift_array[::-1], z)
