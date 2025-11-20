
import os
import jax


import jax
import jax.numpy as jnp
# jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision

import matplotlib.pyplot as plt
from functools import partial 
from jax import random, lax
import blackjax
from functools import partial
import time 
import json
from jax.scipy.special import logsumexp






def run_sharpy(id, folder = 'output/test'):
 

    def prior(params):
        return 0.


    psd = "/leonardo/home/userexternal/gdemasi0/SHARPy-GW/LIGO-P1200087-v18-aLIGO_DESIGN_psd.dat"
    psd = "/Users/gabrieledemasi/dottorato/GIT-repo/SHARPy-GW/sharpy/LIGO-P1200087-v18-aLIGO_DESIGN_psd.dat"

    detector_settings = {
            "H1": {
                "psd_file"  : psd, 
                # "data_file" : '/leonardo/home/userexternal/gdemasi0/SMC/H-H1_GWOSC_4KHZ_R1-1126259447-32.txt',
                # "channel"   : 'GWOSC',
                

                
            },
            "L1": {
                "psd_file"  : psd, 
                # "data_file" : '/leonardo/home/userexternal/gdemasi0/SMC/L-L1_GWOSC_4KHZ_R1-1126259447-32.txt',
                # "channel"   :'GWOSC',
            
            },
                
            
    


                    }


    from sharpy.GW_likelihood import GWNetwork, log_likelihood_det


    truth =  jnp.array([3.0, 1.0, 5.5, jnp.pi/2, jnp.pi, jnp.pi/2, 30.0, 0.7, 0.0, -1, 1.])
    from sharpy.GW_likelihood import GWNetwork, log_likelihood_det
    gw_network = GWNetwork(detector_settings,
                        
                        injection_parameters=truth,
        
                        )

    batched_detector = gw_network.batched_detector

    log_likelihood = partial(log_likelihood_det, detector_list=batched_detector)



   

    from utils import compute_mass_matrix

    def log_posterior(params, beta=1):
        return log_likelihood(params)*beta + prior(params)



    prior_bounds            = jnp.array([[0., 2*jnp.pi], [-jnp.pi/2, jnp.pi/2], [4.9, 8.7], [0., jnp.pi], [0., 2*jnp.pi], [0., jnp.pi], [25, 35], [0.4, 1.], [-1e-1, 1e-1], [-1., 1.], [-1., 1.]])
    boundary_conditions     = jnp.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]) #1: periodic, 0: reflective


    prior_bounds            = jnp.array([[-5, 5] for _ in range(10)])
    boundary_conditions     = jnp.array([0 for _ in range(10)])

    number_of_particles     = 9000
    step_size               = 0.2


    temperature_schedule    = jnp.concatenate((  jnp.array([1e-5]),  jnp.array([1e-4]),jnp.array([1e-3]), jnp.array([5e-3]), jnp.logspace(-2, 0, 30),))

    parameters_names        =  ['ra','dec','logdistance','theta_jn','phiref','pol', 'mc','q', 'tc', 'chi1', 'chi2']


    folder                  = f"{folder}/run_{id}"
    label                   = f"run_{id}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    from smc_functions import run_smc, compute_evidence, draw_iid_samples
    start     = time.time()



    # from test_distributions import bimodal_gaussian_mixture
    # log_likelihood = bimodal_gaussian_mixture(-1., 1., 0.1, 0.5, 10)
    


    samples , samples_dict = run_smc(log_likelihood,
                                                    prior,
                                                    prior_bounds,
                                                    boundary_conditions,
                                                    temperature_schedule,
                                                    number_of_particles,
                                                    step_size,
                                                    master_key=jax.random.PRNGKey(jnp.array(id)),
                                                
                                                    )




    import numpy as np

    logZ, logZerr = compute_evidence( samples_dict)
    samples       = draw_iid_samples(samples_dict,)

    print("logZ = {}, logZerr = {}".format(logZ, logZerr))

    print("the number of samples after rejection sampling is:", len(samples))

    np.savetxt(os.path.join(folder,"posterior_samples.txt"),np.array(samples),)


    from corner import corner
    fig = corner(np.array(samples), 
                show_titles    =True,
                #  truths     = truth,
                labels          = parameters_names, 
                title_kwargs   = {"fontsize": 12},)

    fig.savefig(f"{folder}/{label}_corner.png")



    # print(particles.shape)


    sampling_time = time.time()-start
    print("time = {}".format(sampling_time))

    np.savetxt(os.path.join(folder,"sampling_time.txt"),np.array([sampling_time]))
    np.savetxt(os.path.join(folder,"evidence.txt"),np.array([logZ, logZerr])[np.newaxis], fmt = '%3f')

    import shutil 
    shutil.copy(__file__, os.path.join(folder, "pe_script_copy" + '.py'))



if __name__ == "__main__":

    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--number", default = 0)
    parser.add_argument("--folder", default = 'output/output')

    args=parser.parse_args()

    number = int(args.number)
    folder = args.folder
    run_sharpy(number, folder)