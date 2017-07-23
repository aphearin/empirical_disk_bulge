"""
"""
import numpy as np
from .engines import random_constant_disruption_engine

__all__ = ('random_constant_disruption', )


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


def time_dependent_disruption(num_gals, time_array, **kwargs):
    """
    """
    t1, t2 = kwargs.get('t1', 10), kwargs.get('t2', 14)
    probt1, probt2 = kwargs.get('probt1', 10), kwargs.get('probt2', 14)

    prob_array = np.array([np.interp(t, [t1, t2], [probt1, probt2]) for t in time_array])
    num_times = len(time_array)
    return np.tile(prob_array, num_gals).reshape((num_gals, num_times))


# def mass_dependent_disruption(num_times, logmass_array, **kwargs):
#     """
#     """
#     logm1, logm2 = kwargs.get('logm1', 10), kwargs.get('logm2', 14)
#     probm1, probm2 = kwargs.get('probm1', 10), kwargs.get('probm2', 14)
#     prob_array = np.interp(logmass_array, [logm1, logm2], [probm1, probm2])
#     num_gals = len(logmass_array)
#     return np.repeat(prob_array, num_times).reshape((num_gals, num_times))
