
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial 
from jax import random, lax
from functools import partial
import time 
import json
from jax.scipy.special import logsumexp
import numpy as np





def run_sharpy(id, folder = 'GW150914_evidence'):

   
    
    
    def prior(params):
        return 0.



    data_H1 = "/leonardo/home/userexternal/gdemasi0/SMC/H-H1_GWOSC_4KHZ_R1-1126259447-32.txt"
    data_L1 = "/leonardo/home/userexternal/gdemasi0/SMC/L-L1_GWOSC_4KHZ_R1-1126259447-32.txt"

    detector_settings = {
            "H1": {
                "psd_file"      :  None, 
                "channel"       :  "GWOSC",
                "data_file"     :  data_H1, 
                "duration"      :  2.0,
                
            },
            "L1": {
                "psd_file"  : None, 
                    "channel"   : "GWOSC",
                    "data_file" : data_L1,
                    "duration"  : 2.0,
                
                
            },}


    from sharpy.GW_likelihood import GWNetwork, log_likelihood_det


    truth =  jnp.array([3.0, 1.0, 5.5, jnp.pi/2, jnp.pi, jnp.pi/2, 30.0, 0.7, 0.0, -0.1, 0.1])
    from sharpy.GW_likelihood import GWNetwork, log_likelihood_det

    gw_network = GWNetwork(detector_settings,
                            # injection_parameters=truth,
                            )


    batched_detector        = gw_network.batched_detector
    log_likelihood          = partial(log_likelihood_det, detector_list=batched_detector)


    prior_bounds            = jnp.array([[0., 2*jnp.pi], [-jnp.pi/2, jnp.pi/2], [4.9, 8.7], [0., jnp.pi], [0., 2*jnp.pi], [0., jnp.pi], [25, 35], [0.4, 1.], [-1e-1, 1e-1], [-1., 1.], [-1., 1.]])
    boundary_conditions     = jnp.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]) #1: periodic, 0: reflective

    number_of_particles     = 9000
    step_size               = 0.2
    alpha                   = 0.9

    parameters_names        =  ['ra','dec','logdistance','theta_jn','phiref','pol', 'mc','q', 'tc', 'chi1', 'chi2']

    folder                  = f"{folder}/run_{id}"
    label                   = f"run_{id}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    from sharpy.smc_functions import run_smc

    start     = time.time()





    result_dict = run_smc(  
                            log_likelihood, 
                            prior, 
                            prior_bounds,
                            boundary_conditions, 
                            alpha,
                            number_of_particles, 
                            step_size,   
                            jax.random.PRNGKey(jnp.array(int(id))),
                            folder = ".",
                            label = "run",
                            initial_logZ = 0.0,
                            initial_dlogZ = 0.0,
                                                    )




    samples     = result_dict['posterior_samples']
    np.savetxt(os.path.join(folder,"posterior_samples.txt"),np.array(samples),)

    with open(f"{folder}/{label}_result.json", "w") as f:
        json.dump(result_dict, f,)


    logZ, dlogZ = result_dict['logZ'], result_dict['dlogZ']

    np.savetxt(f"{folder}/{label}_evidence.txt", jnp.array([logZ, dlogZ]) )
                        


    from corner import corner
    fig = corner(np.array(samples), 
                show_titles    = True,
                labels         = parameters_names, 
                title_kwargs   = {"fontsize": 12},)

    fig.savefig(f"{folder}/{label}_corner.png")



    sampling_time = time.time()-start
    print("time = {}".format(sampling_time))
    np.savetxt(f"{folder}/{label}_sampling_time.txt", jnp.array([sampling_time]) )



if __name__ == "__main__":
    
    #add paerse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0, help='ID of the run')
    args = parser.parse_args()
    run_sharpy(args.id)