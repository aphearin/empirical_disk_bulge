"""
"""
import numpy as np
from scipy.stats import gaussian_kde

default_bt_abscissa = np.linspace(0, 1, 20)


def kde_bt_pdf_fixed_mstar_ssfr(log10sm_gals, log10ssfr_gals, bt_gals,
            log10sm_fixed, log10ssfr_fixed, bt_abscissa=default_bt_abscissa):
    """
    """
    dataset = np.array((log10sm_gals, log10ssfr_gals, bt_gals))
    kdeobj = gaussian_kde(dataset)
    bt_pdf = np.array(list(
        kdeobj.evaluate((log10sm_fixed, log10ssfr_fixed, bt))[0] for bt in bt_abscissa))
    return bt_abscissa, bt_pdf


def kde_bt_pdf_fixed_mstar(log10sm_gals, bt_gals, log10sm_fixed, bt_abscissa=default_bt_abscissa):
    """
    """
    dataset = np.array((log10sm_gals, bt_gals))
    kdeobj = gaussian_kde(dataset)
    bt_pdf = np.array(list(
        kdeobj.evaluate((log10sm_fixed, bt))[0] for bt in bt_abscissa))
    return bt_abscissa, bt_pdf
