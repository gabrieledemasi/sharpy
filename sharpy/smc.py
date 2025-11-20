
import os
import jax
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2000"


# This sets the cache directory globally




#uncomment these line if you want to run on cpu
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2000"

import jax
import jax.numpy as jnp
# jax.config.update("jax_enable_x64", False)  # Enable 64-bit precision
# jax.config.update("jax_enable_x64", True) 
import matplotlib.pyplot as plt
from functools import partial 
from jax import random, lax
import blackjax
from functools import partial
import time 
import json



from jax.scipy.special import logsumexp


def run_sharpy(id, global_folder = 'GW150914_evidence'):
 

    def prior(params):
        return 0. # Uniform prior within bounds, log(1) = 0


    psd = "/leonardo/home/userexternal/gdemasi0/SHARPy-GW/LIGO-P1200087-v18-aLIGO_DESIGN_psd.dat"


    detector_settings = {
            "H1": {
                "psd_file"  : None, 
                "data_file" : '/leonardo/home/userexternal/gdemasi0/SMC/H-H1_GWOSC_4KHZ_R1-1126259447-32.txt',
                "channel"   : 'GWOSC',
                
                
            },
            "L1": {
                "psd_file"  : None, 
                "data_file" : '/leonardo/home/userexternal/gdemasi0/SMC/L-L1_GWOSC_4KHZ_R1-1126259447-32.txt',
                "channel"   :'GWOSC',
            
            },
                
            
    


                    }


    from likelihood import GWNetwork, log_likelihood_det



    from likelihood import GWNetwork, log_likelihood_det
    gw_network = GWNetwork(detector_settings,
                        
                        #injection_parameters=truth,
        
                        )

    batched_detector = gw_network.batched_detector

    log_likelihood = partial(log_likelihood_det, detector_list=batched_detector)


    truth =  jnp.array([3.0, 1.0, 5.5, jnp.pi/2, jnp.pi, jnp.pi/2, 30.0, 0.7, 0.0, -1, 1.])
    print(log_likelihood(truth))

    from utils import compute_mass_matrix

    # mass_matrix = compute_mass_matrix(log_likelihood, truth)
    # momentum = jax.random.multivariate_normal(jax.random.PRNGKey(0), jnp.zeros(len(truth)), mass_matrix)
    # print("momentum", momentum)
    # import sys
    # sys.exit()

    def log_posterior(params, beta=1):
        return log_likelihood(params)*beta + prior(params)



    prior_bounds =jnp.array([[0., 2*jnp.pi], [-jnp.pi/2, jnp.pi/2], [4.9, 8.7], [0., jnp.pi], [0., 2*jnp.pi], [0., jnp.pi], [25, 35], [0.4, 1.], [-1e-1, 1e-1], [-1., 1.], [-1., 1.]])






    # prior_bounds            =jnp.array([[0., 2*jnp.pi], [-jnp.pi/2, jnp.pi/2], [3., 6.], [0., jnp.pi], [0., 2*jnp.pi], [0., jnp.pi], [6, 14], [0.4, 1.], [-1e-1, 1e-1]])
    boundary_conditions     = jnp.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])#0: periodic, 1: reflective
    # boundary_conditions     = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])#0: periodic, 1: reflective

    
    number_of_particles     = 9000




    # temperature_schedule    = jnp.concatenate(jnp.logspace(-2, 0, 20),)
    # temperature_schedule    = jnp.logspace(-3, 0, 40)
    # temperature_schedule    = jnp.logspace(-2, 0, 20)
    # temperature_schedule    = jnp.concatenate( jnp.logspace(-2, 0, 30),)
    # temperature_schedule    =jnp.logspace(-1, 0, 10)
    temperature_schedule    = jnp.concatenate((  jnp.array([1e-5]),  jnp.array([1e-4]),jnp.array([1e-3]), jnp.array([5e-3]), jnp.logspace(-2, 0, 30),))


    # temperature_schedule    = temperature_schedule[1:]
    parameters_names        =  ['ra','dec','logdistance','theta_jn','phiref','pol', 'mc','q', 'tc', 'chi1', 'chi2']


    folder                  = f"{global_folder}/run_{id}"
    label                   = f"run_{id}"

    if not os.path.exists(folder):
        os.makedirs(folder)







    from smc_functions import run_smc, run_persistent_smc


    start  = time.time()

    step_size = 0.2



    samples , samples_dict = run_smc(log_likelihood,
                                                    prior,
                                                    prior_bounds,
                                                    boundary_conditions,
                                                    temperature_schedule,
                                                    number_of_particles,
                                                    step_size,
                                                    master_key=jax.random.PRNGKey(jnp.array(id)),
                                                
                                                    )



    # print(particles)
    import numpy as np





    from smc_functions import compute_evidence, draw_iid_samples
    logZ, logZerr = compute_evidence( samples_dict)
    print("logZ = {}, logZerr = {}".format(logZ, logZerr))


    samples = draw_iid_samples(samples_dict,)
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



import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--number")
parser.add_argument("--folder", default = 'GW150914_evidence')

args=parser.parse_args()
if args.number:
    number = int(args.number)
else:
    number = 0


if args.folder:
    folder = args.folder
else:
    folder = 'GW150914_evidence'



run_sharpy(number, folder)