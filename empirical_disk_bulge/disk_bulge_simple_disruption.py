"""
"""
import numpy as np

from .engines import disk_bulge_simple_disruption_engine as disk_bulge_engine
from .engines import disk_in_situ_bulge_ex_situ_engine


__all__ = ('disk_bulge_simple_disruption', )


def disk_bulge_simple_disruption(sfr_history, sm_history,
        disruption_prob_history, cosmic_age_array, redshift_obs, frac_migration_array):
    """
    """
    merger_history = calculate_merger_history(sfr_history, sm_history)
    disruption_prob_history = calculate_disruption_prob_history()

    sm_decomposition = np.array(disk_bulge_engine(
        sfr_history, merger_history, disruption_prob_history,
        frac_migration_array, cosmic_age_array, redshift_obs))
    disk, bulge = sm_decomposition[:, 0], sm_decomposition[:, 1]
    return disk, bulge


def calculate_merger_history(sfr_history, sm_history, cosmic_age_array):
    """
    """
    raise NotImplementedError("Not implemented yet")


def calculate_disruption_prob_history(*args, **kwargs):
    """
    """
    raise NotImplementedError("Not implemented yet")
