import argparse


parser = argparse.ArgumentParser()
parser.add_argument("num_iteration", type=int,
    help="Approximate number of mcmc iterations")
parser.add_argument("-num_burnin", type=int, default=3,
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


from empirical_disk_bulge.models import ssfr_dependent_disruption


def model_prediction(x, sm, ssfr, smh, sfrh, tarr, zobs):
    prob1_disrupt = min(1, max(0, x[0]))
    prob2_disrupt = min(1, max(0, x[1]))
    frac_migration = min(1, max(0, x[2]))
    zobs = 0.

    sm_disk, sm_bulge = ssfr_dependent_disruption(sfrh, smh, tarr, zobs,
        frac_migration, prob1_disrupt, prob2_disrupt, ssfr1=-11.25, ssfr2=-9.)

    bt = sm_bulge/(sm_disk + sm_bulge)

    return bt_measurement(bt, sm, ssfr, return_data_vector=True)


def lnprior(x):
    prob1_disrupt, prob2_disrupt, frac_disrupt = x
    if (0 <= prob1_disrupt <= 1) and (0 <= prob2_disrupt <= 1) and (0 <= frac_disrupt <= 1):
        return 0.0
    else:
        return -np.inf


def lnlike(x, observations, icov, sm, ssfr, smh, sfrh, tarr, zobs):
    predictions = model_prediction(x, sm, ssfr, smh, sfrh, tarr, zobs)
    diff = predictions-observations
    return -np.dot(diff,np.dot(icov,diff))/2.0


def lnprob(x, observations, icov, sm, ssfr, smh, sfrh, tarr, zobs):
    prior = lnprior(x)
    if np.isinf(prior):
        return prior
    else:
        return lnlike(x, observations, icov, sm, ssfr, smh, sfrh, tarr, zobs) + prior


from umachine_pyio.load_mock import value_added_mock, load_mock_from_binaries
subvolumes = np.random.choice(np.arange(144), 50, replace=False)
vamock = value_added_mock(load_mock_from_binaries(subvolumes), 250)
sfrh, smh = vamock['sfr_history_main_prog'].data, vamock['sm_history_main_prog'].data
sm = np.log10(vamock['obs_sm'].data)
ssfr = np.log10(vamock['obs_sfr'].data/vamock['obs_sm'].data)

from umachine_pyio.load_mock import get_snapshot_times
tarr = get_snapshot_times()

zobs = 0.


frac_migration, prob1, prob2 = 0.25, 0.01, 0.1
params = frac_migration, prob1, prob2


import emcee

lnprob_args=[sdss_data_vector, sdss_invcov, sm, ssfr, smh, sfrh, tarr, zobs]
ndim = len(params)

nwalkers = 2*ndim
p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprob_args)

print("...Running burn-in phase of {0} total likelihood evaluations ".format(
    args.num_burnin*nwalkers))
start = time()
pos0, prob, state = sampler.run_mcmc(p0, args.num_burnin)
sampler.reset()
end = time()
print("Total runtime for burn-in = {0:.2f} seconds".format(end-start))

outname = "ssfr_dependent_chain.dat"

sep = "  "
formatter = sep.join("{"+str(i)+":.4f}" for i in range(pos0.shape[-1])) + "  " + "{"+str(pos0.shape[-1])+":.4f}\n"
header = "prob1_disrupt  prob2_disrupt frac_migration  lnprob\n"

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
