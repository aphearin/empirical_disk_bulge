"""
"""
import numpy as np

from .engines import disk_in_situ_bulge_ex_situ_engine


__all__ = ('disk_in_situ_bulge_ex_situ_decomposition', )


def disk_in_situ_bulge_ex_situ_decomposition(sfr_history, sm_history, cosmic_age_array, zobs):
    """
    Examples
    --------
    >>> ngals, ntimes = 100, 178
    >>> sfr_history = np.random.random((ngals, ntimes))
    >>> merger_history = np.random.random((ngals, ntimes))
    >>> cosmic_age_array = np.linspace(0.1, 14, ntimes)
    >>> zobs = 0.1
    >>> sm_disk, sm_bulge = disk_in_situ_bulge_ex_situ_decomposition(sfr_history, merger_history, cosmic_age_array, zobs)
    """
    dsm_history = np.insert(np.diff(sm_history), 0, sm_history[:, 0], axis=1)
    disk_bulge_result = np.array(
        disk_in_situ_bulge_ex_situ_engine(sfr_history, dsm_history, cosmic_age_array, zobs))
    sm_disk, sm_bulge = disk_bulge_result[:, 0], disk_bulge_result[:, 1]
    return sm_disk, sm_bulge

