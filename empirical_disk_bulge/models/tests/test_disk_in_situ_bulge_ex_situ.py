"""
"""
import numpy as np
from umachine_pyio.load_mock import get_snapshot_times, load_mock_from_binaries, value_added_mock

from ..engines import disk_in_situ_bulge_ex_situ_engine

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
