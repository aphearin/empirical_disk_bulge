"""
"""
cimport cython
import numpy as np
cimport numpy as cnp
from ..sfr_integration import (_stellar_mass_integrand_factors,
        _index_of_nearest_larger_redshift, bolplanck_redshifts)


__all__ = ('disk_in_situ_bulge_ex_situ_engine', )



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def disk_in_situ_bulge_ex_situ_engine(double[:, :] in_situ_sfr_history,
        double[:, :] merging_history, cosmic_age_array, redshift_obs):
    """
    """
    #  dt_arr stores the array of time steps
    #  frac_remaining_arr the mass loss fraction due to stellar evolution
    idx_zobs = _index_of_nearest_larger_redshift(redshift_obs, bolplanck_redshifts)
    _dt, _frac_remaining = _stellar_mass_integrand_factors(
            cosmic_age_array[idx_zobs], cosmic_age_array)
    cdef double[:] dt_arr = np.array(_dt, dtype='f8', order='C')
    cdef double dt
    cdef double[:] frac_remaining_arr = np.array(_frac_remaining, dtype='f8', order='C')
    cdef double frac_remaining

    #  Determine shape of the star-formation histories
    cdef int num_gals = in_situ_sfr_history.shape[0]
    cdef int num_time_steps = in_situ_sfr_history.shape[1]

    #  Declare output array and loop variables
    cdef double[:, :] disk_bulge_result = np.zeros((num_gals, 2), dtype='f8', order='C')
    cdef double sm_bulge, sm_disk
    cdef int igal, itime

    #  Declare variables needed by the model
    cdef double merging_dsm, in_situ_dsm

    #  Outer loop is over rows, one for each galaxy in the mock
    for igal in range(num_gals):
        sm_bulge = 0.
        sm_disk = 0.

        #  Inner loop is over columns, one for every timestep
        for itime in range(num_time_steps):

            #  Retrieve the amount of time that has passed between snapshots
            dt = dt_arr[itime]

            #  Retrieve the (pre-computed) amount of fractional stellar mass loss
            frac_remaining = frac_remaining_arr[itime]

            #  Add all in-situ SFR into the disk
            in_situ_dsm = in_situ_sfr_history[igal, itime]*dt*frac_remaining*1e9
            sm_disk += in_situ_dsm

            #  Add all mergers into the bulge
            merging_dsm = merging_history[igal, itime]
            sm_bulge += merging_dsm

        #  Update the output arrays before moving on to the next galaxy
        disk_bulge_result[igal, 0] = sm_disk
        disk_bulge_result[igal, 1] = sm_bulge

    return disk_bulge_result
