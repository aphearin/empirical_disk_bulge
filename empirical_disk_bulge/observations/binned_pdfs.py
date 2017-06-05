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
    prob : float or ndarray
        Float or ndarray of shape (ngals, ) storing
        the probability that the input galaxy will be selected by the filtering kernel

    Examples
    ---------
    >>> ngals = int(1e4)
    >>> stellar_mass = np.random.uniform(9.5, 11.75, ngals)
    >>> mu, sigma = 10.3, 0.1
    >>> selection_prob = gaussian_kernel_selection(stellar_mass, mu, sigma)
    >>> selection_mask = np.random.rand(ngals) < selection_prob
    """
    return 1 - 2*np.abs(0.5 - norm.cdf(x, loc=mu, scale=sigma))
