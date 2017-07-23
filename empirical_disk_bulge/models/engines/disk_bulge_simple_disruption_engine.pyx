"""
"""
cimport cython
import numpy as np
cimport numpy as cnp
from ..sfr_integration import (_stellar_mass_integrand_factors,
        _index_of_nearest_larger_redshift, bolplanck_redshifts)
from libc.math cimport fmax as c_fmax
from libc.stdlib cimport rand as c_rand
from libc.stdlib cimport RAND_MAX


cdef double random_uniform():
    cdef double r = c_rand()
    return r/RAND_MAX

__all__ = ('simple_disruption_engine', )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def simple_disruption_engine(double[:, :] in_situ_sfr_history,
        double[:, :] dsm_main_prog, double[:, :] prob_disrupt_history,
        cosmic_age_array, redshift_obs, double frac_migration):
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
    cdef double sm_bulge, sm_disk, dsm_mergers
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
            in_situ_dsm = in_situ_sfr_history[igal, itime]*dt*1e9
            sm_disk += in_situ_dsm*frac_remaining

            #  Add all mergers into the bulge
            dsm_mergers = dsm_main_prog[igal, itime] - in_situ_dsm
            sm_bulge += c_fmax(0., dsm_mergers*frac_remaining)

            #  Disrupt the disk according to the input probability
            if random_uniform() < prob_disrupt_history[igal, itime]*dt:
                disk_to_bulge_migration_mass = frac_migration*sm_disk
                sm_bulge += disk_to_bulge_migration_mass
                sm_disk -= disk_to_bulge_migration_mass


        #  Update the output arrays before moving on to the next galaxy
        disk_bulge_result[igal, 0] = sm_disk
        disk_bulge_result[igal, 1] = sm_bulge

    return disk_bulge_result
