"""
"""
import numpy as np
from scipy.stats import norm


def gaussian_kernel_selection(x, mu, sigma):
    """ Probability that a point ``x`` should be selected according to
    a Gaussian kernel with input mean ``mu`` and width ``sigma``.

    Parameters
    ----------
    x : float or ndarray
        Float or ndarray of shape (ngals, ) storing
        the attribute of the data determining the probability of selection

    mu : float
        Center of the selection function

    sigma : float
        Gaussian spread of the selection function

    Returns
    -------
    mask : float or ndarray
        Boolean array of shape (ngals, ) that can be used as a mask to select
        values of the input ``x`` passing the kernel selection

    Examples
    ---------
    >>> ngals = int(1e4)
    >>> stellar_mass = np.random.uniform(9.5, 11.75, ngals)
    >>> mu, sigma = 10.3, 0.1
    >>> selection_prob = gaussian_kernel_selection(stellar_mass, mu, sigma)
    >>> selection_mask = np.random.rand(ngals) < selection_prob
    """
    prob = 1 - 2*np.abs(0.5 - norm.cdf(x, loc=mu, scale=sigma))
    return np.random.rand(len(x)) < prob


def bt_pdf_fixed_sm(bt, sm, fixed_sm, bt_bins, sigma=0.1):
    """
    """
    sample_mask = gaussian_kernel_selection(sm, fixed_sm, sigma=sigma)
    assert np.count_nonzero(sample_mask) > 100, "Must have at least 100 galaxies in bin to estimate PDF"
    return np.histogram(bt[sample_mask], bt_bins, density=True)[0]


def bt_pdf_fixed_sm_ssfr(bt, sm, ssfr, fixed_sm, fixed_ssfr, bt_bins,
            sigma_sm=0.1, sigma_ssfr=0.25):
    """
    """
    sm_sample_mask = gaussian_kernel_selection(sm, fixed_sm, sigma=sigma_sm)
    ssfr_mask = gaussian_kernel_selection(ssfr[sm_sample_mask], fixed_ssfr, sigma=sigma_ssfr)
    assert np.count_nonzero(ssfr_mask) > 100, "Must have at least 100 galaxies in bin to estimate PDF"
    return np.histogram(bt[sm_sample_mask][ssfr_mask], bt_bins, density=True)[0]


def bulge_disk_fractions_vs_sm(bt, sm, sm_abscissa, domination_vals, sigma_sm):
    """
    """
    frac_disk_dom = np.zeros_like(sm_abscissa)
    frac_bulge_dom = np.zeros_like(sm_abscissa)

    for i, sm_mid in enumerate(sm_abscissa):
        sm_mask = gaussian_kernel_selection(sm, sm_mid, sigma_sm)
        nbin_total = np.count_nonzero(sm_mask)
        nbin_disk_dom = np.count_nonzero(bt[sm_mask] < domination_vals[0])
        nbin_bulge_dom = np.count_nonzero(bt[sm_mask] > domination_vals[1])

        frac_disk_dom[i] = nbin_disk_dom/float(nbin_total)
        frac_bulge_dom[i] = nbin_bulge_dom/float(nbin_total)

    return frac_disk_dom, frac_bulge_dom


def sfr_sequence_bt_classification_vs_sm(bt, sm, ssfr,
            sm_abscissa=np.arange(9.75, 11.35, 0.1), domination_vals=(0.25, 0.75),
            sigma_sm=0.1, gv_range=(-11.25, -10.75)):
    """
    """
    sfs_range = (gv_range[1], np.inf)
    q_range = (-np.inf, gv_range[0])

    sfs_mask = (ssfr >= sfs_range[0]) & (ssfr < sfs_range[1])
    gv_mask = (ssfr >= gv_range[0]) & (ssfr < gv_range[1])
    q_mask = (ssfr >= q_range[0]) & (ssfr < q_range[1])

    sfs_bt = bt[sfs_mask]
    gv_bt = bt[gv_mask]
    q_bt = bt[q_mask]
    sfs_sm = sm[sfs_mask]
    gv_sm = sm[gv_mask]
    q_sm = sm[q_mask]

    frac_disk_dom_all, frac_bulge_dom_all = bulge_disk_fractions_vs_sm(
            bt, sm, sm_abscissa, domination_vals, sigma_sm)
    frac_disk_dom_sfs, frac_bulge_dom_sfs = bulge_disk_fractions_vs_sm(
            sfs_bt, sfs_sm, sm_abscissa, domination_vals, sigma_sm)
    frac_disk_dom_gv, frac_bulge_dom_gv = bulge_disk_fractions_vs_sm(
            gv_bt, gv_sm, sm_abscissa, domination_vals, sigma_sm)
    frac_disk_dom_q, frac_bulge_dom_q = bulge_disk_fractions_vs_sm(
            q_bt, q_sm, sm_abscissa, domination_vals, sigma_sm)

    return (sm_abscissa, frac_disk_dom_all, frac_bulge_dom_all,
            frac_disk_dom_sfs, frac_bulge_dom_sfs,
            frac_disk_dom_gv, frac_bulge_dom_gv,
            frac_disk_dom_q, frac_bulge_dom_q)
