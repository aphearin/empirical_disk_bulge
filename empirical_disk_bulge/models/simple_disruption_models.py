"""
"""
import numpy as np
from scipy.special import erf

from .engines import (random_constant_disruption_engine, simple_disruption_engine,
        merger_triggered_disruption_engine)

__all__ = ('random_constant_disruption', 'time_dependent_disruption',
    'ssfr_dependent_disruption', 'merger_triggered_disruption')


def random_constant_disruption(sfr_history, sm_history, cosmic_age_array, zobs,
        disruption_prob, frac_migration):
    """
    Examples
    --------
    >>> ngals, ntimes = 100, 178
    >>> sfr_history = np.random.random((ngals, ntimes))
    >>> merger_history = np.random.random((ngals, ntimes))
    >>> cosmic_age_array = np.linspace(0.1, 14, ntimes)
    >>> zobs = 0.1
    >>> disruption_prob, frac_migration = 0.02, 0.25
    >>> sm_disk, sm_bulge = random_constant_disruption(sfr_history, merger_history, cosmic_age_array, zobs, disruption_prob, frac_migration)
    """
    dsm_history = np.insert(np.diff(sm_history), 0, sm_history[:, 0], axis=1)
    disk_bulge_result = np.array(
        random_constant_disruption_engine(sfr_history, dsm_history, cosmic_age_array, zobs,
                        disruption_prob, frac_migration))
    sm_disk, sm_bulge = disk_bulge_result[:, 0], disk_bulge_result[:, 1]
    return sm_disk, sm_bulge


def time_dependent_disruption(sfr_history, sm_history, cosmic_age_array, zobs,
        frac_migration, prob1, prob2, t1=1.5, t2=13.8, return_disruption_history=False):
    """
    Examples
    --------
    >>> ngals, ntimes = 100, 178
    >>> sfr_history = np.random.random((ngals, ntimes))
    >>> sm_history = np.random.random((ngals, ntimes))
    >>> cosmic_age_array = np.linspace(0.1, 14, ntimes)
    >>> zobs = 0.1
    >>> frac_migration = 0.25
    >>> prob1, prob2 = 0.05, 0.01
    >>> sm_disk, sm_bulge = time_dependent_disruption(sfr_history, sm_history, cosmic_age_array, zobs, frac_migration, prob1, prob2)
    """
    num_gals = np.shape(sfr_history)[0]
    prob_array = np.array([np.interp(t, [t1, t2], [prob1, prob2]) for t in cosmic_age_array])
    num_times = len(cosmic_age_array)
    prob_disrupt_history = np.tile(prob_array, num_gals).reshape((num_gals, num_times))

    dsm_history = np.insert(np.diff(sm_history), 0, sm_history[:, 0], axis=1)

    _engine_output = simple_disruption_engine(sfr_history, dsm_history, prob_disrupt_history,
                cosmic_age_array, zobs, frac_migration,
                return_disruption_history=return_disruption_history)
    if return_disruption_history:
        disk_bulge_decomposition, disruption_history = _engine_output
        disk_bulge_array = np.array(disk_bulge_decomposition)
        sm_disk, sm_bulge = disk_bulge_array[:, 0], disk_bulge_array[:, 1]
        return sm_disk, sm_bulge, np.array(disruption_history)
    else:
        disk_bulge_array = np.array(_engine_output)
        sm_disk, sm_bulge = disk_bulge_array[:, 0], disk_bulge_array[:, 1]
        return sm_disk, sm_bulge


def sm_dependent_disruption(sfr_history, sm_history, cosmic_age_array, zobs,
        frac_migration, prob1, prob2, logsm1=9, logsm2=11.25):
    """
    Examples
    --------
    >>> ngals, ntimes = 100, 178
    >>> sfr_history = np.random.random((ngals, ntimes))
    >>> sm_history = np.random.random((ngals, ntimes))
    >>> cosmic_age_array = np.linspace(0.1, 14, ntimes)
    >>> zobs = 0.1
    >>> frac_migration = 0.25
    >>> prob1, prob2 = 0.05, 0.01
    >>> sm_disk, sm_bulge = sm_dependent_disruption(sfr_history, sm_history, cosmic_age_array, zobs, frac_migration, prob1, prob2)
    """
    prob_disrupt_history = np.interp(sm_history, [logsm1, logsm2], [prob1, prob2])

    dsm_history = np.insert(np.diff(sm_history), 0, sm_history[:, 0], axis=1)

    disk_bulge_result = np.array(
        simple_disruption_engine(sfr_history, dsm_history, prob_disrupt_history,
                cosmic_age_array, zobs, frac_migration))
    sm_disk, sm_bulge = disk_bulge_result[:, 0], disk_bulge_result[:, 1]
    return sm_disk, sm_bulge


def ssfr_dependent_disruption(sfr_history, sm_history, cosmic_age_array, zobs,
        frac_migration, prob1, prob2, ssfr1=-11.25, ssfr2=-9., return_disruption_history=False):
    """
    Examples
    --------
    >>> ngals, ntimes = 100, 178
    >>> sfr_history = np.random.random((ngals, ntimes))
    >>> sm_history = np.random.random((ngals, ntimes))
    >>> cosmic_age_array = np.linspace(0.1, 14, ntimes)
    >>> zobs = 0.1
    >>> frac_migration = 0.25
    >>> prob1, prob2 = 0.05, 0.01
    >>> sm_disk, sm_bulge = ssfr_dependent_disruption(sfr_history, sm_history, cosmic_age_array, zobs, frac_migration, prob1, prob2)
    """
    ssfr_history = np.where(sm_history == 0, -np.inf, np.log10(sfr_history/sm_history))
    prob_disrupt_history = np.interp(ssfr_history, [ssfr1, ssfr2], [prob1, prob2])

    dsm_history = np.insert(np.diff(sm_history), 0, sm_history[:, 0], axis=1)

    _engine_output = simple_disruption_engine(sfr_history, dsm_history, prob_disrupt_history,
                cosmic_age_array, zobs, frac_migration,
                return_disruption_history=return_disruption_history)

    if return_disruption_history:
        disk_bulge_decomposition, disruption_history = _engine_output
        disk_bulge_array = np.array(disk_bulge_decomposition)
        sm_disk, sm_bulge = disk_bulge_array[:, 0], disk_bulge_array[:, 1]
        return sm_disk, sm_bulge, np.array(disruption_history)
    else:
        disk_bulge_array = np.array(_engine_output)
        sm_disk, sm_bulge = disk_bulge_array[:, 0], disk_bulge_array[:, 1]
        return sm_disk, sm_bulge


def _zscore_to_percentile(z):
    return 0.5*(1 + erf(z/np.sqrt(2)))


def merger_triggered_disruption(sfr_history, sm_history, merger_threshold_history,
        cosmic_age_array, zobs, frac_migration, return_disruption_history=False):
    """
    Examples
    --------
    >>> ngals, ntimes = 100, 178
    >>> sfr_history = np.random.random((ngals, ntimes))
    >>> sm_history = np.random.random((ngals, ntimes))
    >>> merger_threshold_history = np.random.random((ngals, ntimes))
    >>> cosmic_age_array = np.linspace(0.1, 14, ntimes)
    >>> zobs = 0.1
    >>> frac_migration = 0.25
    >>> sm_disk, sm_bulge = merger_triggered_disruption(sfr_history, sm_history, merger_threshold_history, cosmic_age_array, zobs, frac_migration)
    """

    dsm_history = np.insert(np.diff(sm_history), 0, sm_history[:, 0], axis=1)

    _engine_output = merger_triggered_disruption_engine(
            sfr_history, dsm_history, merger_threshold_history,
                cosmic_age_array, zobs, frac_migration,
                return_disruption_history=return_disruption_history)

    if return_disruption_history:
        disk_bulge_decomposition, disruption_history = _engine_output
        disk_bulge_array = np.array(disk_bulge_decomposition)
        sm_disk, sm_bulge = disk_bulge_array[:, 0], disk_bulge_array[:, 1]
        return sm_disk, sm_bulge, np.array(disruption_history)
    else:
        disk_bulge_array = np.array(_engine_output)
        sm_disk, sm_bulge = disk_bulge_array[:, 0], disk_bulge_array[:, 1]
        return sm_disk, sm_bulge

