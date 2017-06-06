"""
"""
import numpy as np
from scipy.stats import norm


def gaussian_kernel_selection(x, mu, sigma):
    """ Return a the probability that a point ``x`` should be selected according to
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
