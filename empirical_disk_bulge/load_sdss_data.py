import os
import numpy as np
from astropy.table import Table


__all__ = ('load_umachine_sdss_complete', )


def load_umachine_sdss_complete(dirname='/Users/aphearin/Dropbox/bt_model/data',
        zmin=0.02, zmax=0.1, sm_low=9.75, sm_high=11.75):

    basename = 'umachine_sdss_dr10_value_added_bt.hdf5'
    fname = os.path.join(dirname, basename)
    full_sdss = Table.read(fname, path='data')

    mask = (full_sdss['z'] > zmin) & (full_sdss['z'] < zmax)
    mask *= full_sdss['sm'] > sm_low
    mask *= full_sdss['sm'] < sm_high
    mask *= ~np.isnan(full_sdss['logMB_mendel13'])
    mask *= ~np.isnan(full_sdss['logMD_mendel13'])
    sdss = full_sdss[mask]

    sdss['ssfr'] = np.log10(sdss['sfr'] / 10**sdss['sm'])
    sdss['bt'] = (10**sdss['logMB_mendel13'])/(10**sdss['logMB_mendel13']+10**sdss['logMD_mendel13'])

    completeness_table = np.loadtxt(os.path.join(dirname, 'completeness.dat'))
    completeness_limit = np.interp(sdss['sm'], completeness_table[:, 0], completeness_table[:, 1])
    is_complete = sdss['z'] < completeness_limit

    return sdss[is_complete]
