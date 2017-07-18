"""
"""
import numpy as np
from umachine_pyio.load_mock import (get_snapshot_times,
        load_mock_from_binaries, value_added_mock)

from ..engines import disk_in_situ_bulge_ex_situ_engine
from ..sfr_integration import in_situ_fraction, bolplanck_redshifts


__all__ = ('test_disk_in_situ_bulge_ex_situ_engine', )

subvolumes = np.random.choice(range(144), 15, replace=False)
va_mock = value_added_mock(load_mock_from_binaries(subvolumes=subvolumes), 250.)


def test_disk_in_situ_bulge_ex_situ_engine():
    in_situ_sfr_history = va_mock['sfr_history_main_prog']
    merging_history = np.zeros_like(in_situ_sfr_history)
    cosmic_age_array = get_snapshot_times()
    redshift_obs = 0.
    result = np.array(disk_in_situ_bulge_ex_situ_engine(in_situ_sfr_history,
        merging_history, cosmic_age_array, redshift_obs))
    assert np.shape(result) == (in_situ_sfr_history.shape[0], 2)


def test_in_situ_fraction_comparison():
    cosmic_age_array = get_snapshot_times()
    sfr_history = va_mock['sfr_history_main_prog']
    sm_mp_history = va_mock['sm_history_main_prog']
    redshift_obs = 0.

    frac_in_situ = in_situ_fraction(sfr_history, sm_mp_history, redshift_obs, cosmic_age_array)
    assert np.all(frac_in_situ >= 0)
    assert np.all(frac_in_situ <= 1.1)

    merging_history = np.zeros_like(sfr_history)
    result = np.array(disk_in_situ_bulge_ex_situ_engine(sfr_history,
            merging_history, cosmic_age_array, redshift_obs))
    sm_disk, sm_bulge = result[:, 0], result[:, 1]
    assert np.all(sm_bulge == 0)

    frac_disk = sm_disk / va_mock['sm']
    assert np.allclose(frac_disk, frac_in_situ, rtol=0.3)
    assert np.allclose(frac_disk, frac_in_situ, rtol=0.1)

