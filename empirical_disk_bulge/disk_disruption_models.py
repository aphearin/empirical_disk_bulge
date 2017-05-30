"""
"""
import numpy as np


__all__ = ('constant_disruption', )


def constant_disruption(num_gals, num_snapshots, **kwargs):
    """
    """
    constant_disruption_prob = kwargs.get('constant_disruption_prob', 0.)
    return np.zeros((num_gals, num_snapshots), dtype='f8') + constant_disruption_prob


