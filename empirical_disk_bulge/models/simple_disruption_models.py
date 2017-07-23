"""
"""
import numpy as np
from .engines import random_constant_disruption_engine, simple_disruption_engine

__all__ = ('random_constant_disruption', 'time_dependent_disruption')


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
        frac_migration, prob1, prob2, t1=1.5, t2=13.8):
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

    disk_bulge_result = np.array(
        simple_disruption_engine(sfr_history, dsm_history, prob_disrupt_history,
                cosmic_age_array, zobs, frac_migration))
    sm_disk, sm_bulge = disk_bulge_result[:, 0], disk_bulge_result[:, 1]
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


# def mass_dependent_disruption(num_times, logmass_array, **kwargs):
#     """
#     """
#     logm1, logm2 = kwargs.get('logm1', 10), kwargs.get('logm2', 14)
#     probm1, probm2 = kwargs.get('probm1', 10), kwargs.get('probm2', 14)
#     prob_array = np.interp(logmass_array, [logm1, logm2], [probm1, probm2])
#     num_gals = len(logmass_array)
#     return np.repeat(prob_array, num_times).reshape((num_gals, num_times))
