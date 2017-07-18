"""
"""
import numpy as np

from .engines import disk_in_situ_bulge_ex_situ_engine


__all__ = ('bt_disk_in_situ_bulge_ex_situ', )


def bt_disk_in_situ_bulge_ex_situ(sfr_history, merger_history, cosmic_age_array, zobs):
    """
    Examples
    --------
    >>> ngals, ntimes = 100, 178
    >>> sfr_history = np.random.random((ngals, ntimes))
    >>> merger_history = np.random.random((ngals, ntimes))
    >>> cosmic_age_array = np.linspace(0.1, 14, ntimes)
    >>> zobs = 0.1
    >>> result = bt_disk_in_situ_bulge_ex_situ(sfr_history, merger_history, cosmic_age_array, zobs)
    """
    sm_decomposition = np.array(
        disk_in_situ_bulge_ex_situ_engine(sfr_history, merger_history, cosmic_age_array, zobs))
    sm_disk, sm_bulge = sm_decomposition[:, 0], sm_decomposition[:, 1]
    return sm_bulge/(sm_disk+sm_bulge)

