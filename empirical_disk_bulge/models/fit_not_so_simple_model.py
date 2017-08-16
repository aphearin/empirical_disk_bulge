import argparse


parser = argparse.ArgumentParser()
parser.add_argument("num_iteration", type=int,
    help="Approximate number of mcmc iterations")
parser.add_argument("-num_burnin", type=int, default=10,
    help="Number of steps per walker for the burn-in")

args = parser.parse_args()


from time import time
import numpy as np
from empirical_disk_bulge.observations import load_umachine_sdss_complete

sdss = load_umachine_sdss_complete()
mask = sdss['type_mendel13'] != 4
mask *= sdss['deltaBD_mendel13'] <= 1
cut_sdss = sdss[mask]

from empirical_disk_bulge.observations import sfr_sequence_bulge_disk_fractions_vs_sm as bt_measurement
_result = bt_measurement(cut_sdss['bt'], cut_sdss['sm'], cut_sdss['ssfr'])
np.save('frac_bulge_disk_vs_sm_mendel13', np.array(_result))
sm_abscissa, frac_disk_dom_all, frac_bulge_dom_all,    frac_disk_dom_sfs, frac_bulge_dom_sfs,     frac_disk_dom_gv, frac_bulge_dom_gv,     frac_disk_dom_q, frac_bulge_dom_q = _result

sdss_data_vector = bt_measurement(cut_sdss['bt'], cut_sdss['sm'], cut_sdss['ssfr'], return_data_vector=True)
sdss_invcov = np.diag(0.2*sdss_data_vector)


from empirical_disk_bulge.models import ssfr_dependent_disruption2


def model_prediction(params, sm, ssfr, smh, sfrh, mh, tarr, noise_level):
    frac_migration1, frac_migration2, logmhcrit, prob1, prob2, ssfr1, ssfr2 = params
    zobs = 0.

    sm_disk, sm_bulge = ssfr_dependent_disruption2(sfrh, smh, mh,
        tarr, zobs, frac_migration1, frac_migration2, 10**logmhcrit, prob1, prob2,
        ssfr1=ssfr1, ssfr2=ssfr2, return_disruption_history=False)

    bt = sm_bulge/(sm_disk + sm_bulge)

    return bt_measurement(bt, sm, ssfr, return_data_vector=True)


def lnprior(params):
    frac_migration1, frac_migration2, logmhcrit, prob1, prob2, ssfr1, ssfr2 = params
    frac_migration1_mask = 0 <= frac_migration1 <= 1
    frac_migration2_mask = 0 <= frac_migration2 <= 1
    logmhcrit_mask = (12.25 <= logmhcrit <= 12.75)
    prob1_mask = (0 <= prob1 <= 1)
    prob2_mask = (0 <= prob2 <= 1)
    ssfr1_mask = (-11.5 <= ssfr1 <= -10.5)
    ssfr2_mask = (-10 <= ssfr2 <= -9)

    viable_mask = (frac_migration1_mask & frac_migration2_mask & logmhcrit_mask
        & prob1_mask & prob2_mask & ssfr1_mask & ssfr2_mask)

    if viable_mask:
        return 0.0
    else:
        return -np.inf


def lnlike(params, observations, icov, sm, ssfr, smh, mh, sfrh, tarr, zobs):
    predictions = model_prediction(params, sm, ssfr, smh, sfrh, mh, tarr, 0.1)
    diff = predictions-observations
    return -np.dot(diff, np.dot(icov, diff))/2.0


def lnprob(params, observations, icov, sm, ssfr, smh, mh, sfrh, tarr, zobs):
    prior = lnprior(params)
    if np.isinf(prior):
        return prior
    else:
        return lnlike(params, observations, icov, sm, ssfr, smh, mh, sfrh, tarr, zobs) + prior


from umachine_pyio.load_mock import value_added_mock, load_mock_from_binaries
subvolumes = np.random.choice(np.arange(144), 50, replace=False)
galprops = list(('upid', 'obs_sfr', 'icl', 'mpeak_history_main_prog', 'sfr_history_main_prog', 'mpeak', 'sfr', 'halo_id',
                'mvir', 'rvir', 'vx', 'sm', 'vy', 'y', 'x', 'sm_history_main_prog', 'z', 'vz', 'obs_sm'))
vamock = value_added_mock(load_mock_from_binaries(subvolumes, galprops=galprops), 250)
vamock['ssfr'] = np.log10(vamock['obs_sfr'].data/vamock['obs_sm'].data)

from umachine_pyio.load_mock import get_snapshot_times
cosmic_age_array = get_snapshot_times()

redshift_zero = 0.


frac_migration1, frac_migration2, logmhcrit, prob1, prob2, ssfr1, ssfr2 = 0.5, 0.1, 12.5, 0.5, 0.1, -11, -9.5
params = frac_migration1, frac_migration2, logmhcrit, prob1, prob2, ssfr1, ssfr2
ndim = len(params)

nwalkers = 2*ndim
p0 = np.zeros((nwalkers, ndim))
p0[:, 0] = np.random.uniform(0, 1, nwalkers)
p0[:, 1] = np.random.uniform(0, 1, nwalkers)
p0[:, 2] = np.random.uniform(12, 13, nwalkers)
p0[:, 3] = np.random.uniform(0, 1, nwalkers)
p0[:, 4] = np.random.uniform(0, 1, nwalkers)
p0[:, 5] = np.random.uniform(-11.25, -10.75, nwalkers)
p0[:, 6] = np.random.uniform(-10, -9, nwalkers)

from emcee import EnsembleSampler

lnprob_args = list((sdss_data_vector, sdss_invcov,
    np.log10(vamock['obs_sm'].data), vamock['ssfr'].data,
    vamock['sm_history_main_prog'].data, vamock['sfr_history_main_prog'].data,
    vamock['mpeak_history_main_prog'].data, cosmic_age_array, redshift_zero))

sampler = EnsembleSampler(nwalkers, ndim, lnprob, args=lnprob_args)

print("...Running burn-in phase of {0} total likelihood evaluations ".format(
    args.num_burnin*nwalkers))
start = time()
pos0, prob, state = sampler.run_mcmc(p0, args.num_burnin)
sampler.reset()
end = time()
print("Total runtime for burn-in = {0:.2f} seconds".format(end-start))






outname = "toy_complicated_chain.dat"

sep = "  "
formatter = sep.join("{"+str(i)+":.4f}" for i in range(pos0.shape[-1])) + "  " + "{"+str(pos0.shape[-1])+":.4f}\n"
header = "frac_migration1  frac_migration2  logmhcrit  prob1  prob2  ssfr1  ssfr2  lnprob\n"

start = time()

print("...Running MCMC with {0} chain elements".format(args.num_iteration*nwalkers))
with open(outname, "wb") as f:
    f.write(header)
    for result in sampler.sample(pos0, iterations=args.num_iteration, storechain=False):
        pos, prob, state = result
        for a, b in zip(pos, prob):
            newline = formatter.format(*np.append(a, b))
            f.write(newline)
end = time()
print("Runtime for MCMC = {0:.2f} minutes".format((end-start)/60.))
print("\a\a\a")

from astropy.table import Table
chain = Table.read(outname, format='ascii')
print("Successfully loaded chain with {0} elements from disk after completion of MCMC".format(len(chain)))
