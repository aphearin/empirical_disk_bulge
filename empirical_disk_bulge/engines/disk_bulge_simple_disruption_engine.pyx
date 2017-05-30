"""
"""
cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand, RAND_MAX
from ..sfr_integration import (_stellar_mass_integrand_factors,
    _index_of_nearest_larger_redshift)

__all__ = ('disk_bulge_simple_disruption_engine', )


cdef double random_uniform():
    cdef double r = rand()
    return r/RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def disk_bulge_simple_disruption_engine(double[:, :] in_situ_sfr_history,
        double[:, :] merging_history, double[:, :] disruption_prob_history,
        double[:] frac_migration_array, cosmic_age_array, redshift_obs):
    """
    """
    cdef int idx_zobs = _index_of_nearest_larger_redshift(redshift_obs)
    _dt, _frac_remaining = _stellar_mass_integrand_factors(
            cosmic_age_array[idx_zobs], cosmic_age_array)
    cdef double[:] dt_arr = np.array(_dt, dtype='f8', order='C')
    cdef double[:] frac_remaining_arr = np.array(_frac_remaining, dtype='f8', order='C')
    cdef double dt, frac_remaining

    cdef int num_gals = in_situ_sfr_history.shape[0]
    cdef int num_time_steps = in_situ_sfr_history.shape[1]

    cdef double[:, :] sm_decomposition = np.zeros((num_gals, 2), dtype='f8', order='C')

    cdef int igal, itime
    cdef double sm_bulge, sm_disk, merging_dsm, disruption_prob
    cdef double disk_to_bulge_migration_mass, in_situ_dsm
    cdef double frac_migration

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

            #  Disrupt the disk according to the input probability
            disruption_prob = disruption_prob_history[igal, itime]
            frac_migration = frac_migration_array[itime]
            if random_uniform() < disruption_prob:
                disk_to_bulge_migration_mass = frac_migration*sm_disk
                sm_bulge += disk_to_bulge_migration_mass
                sm_disk -= disk_to_bulge_migration_mass

        #  Update the output arrays before moving on to the next galaxy
        sm_decomposition[igal, 0] = sm_disk
        sm_decomposition[igal, 1] = sm_bulge

    return sm_decomposition
