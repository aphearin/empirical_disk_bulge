"""
"""
import numpy as np


__all__ = ('constant_disruption', )


def constant_disruption(num_gals, num_snapshots, **kwargs):
    """
    """
    constant_disruption_prob = kwargs.get('constant_disruption_prob', 0.)
    return np.zeros((num_gals, num_snapshots), dtype='f8') + constant_disruption_prob


def time_dependent_disruption(num_gals, time_array, **kwargs):
    """
    """
    t1, t2 = kwargs.get('t1', 10), kwargs.get('t2', 14)
    probt1, probt2 = kwargs.get('probt1', 10), kwargs.get('probt2', 14)

    prob_array = np.array([np.interp(t, [t1, t2], [probt1, probt2]) for t in time_array])
    num_times = len(time_array)
    return np.tile(prob_array, num_gals).reshape((num_gals, num_times))


def mass_dependent_disruption(num_times, logmass_array, **kwargs):
    """
    """
    logm1, logm2 = kwargs.get('logm1', 10), kwargs.get('logm2', 14)
    probm1, probm2 = kwargs.get('probm1', 10), kwargs.get('probm2', 14)
    prob_array = np.interp(logmass_array, [logm1, logm2], [probm1, probm2])
    num_gals = len(logmass_array)
    return np.repeat(prob_array, num_times).reshape((num_gals, num_times))
