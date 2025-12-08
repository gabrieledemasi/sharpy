
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial 
from jax import random, lax
import blackjax
from functools import partial
import time 
import json
from jax.scipy.special import logsumexp
import numpy as np
import sharpy
import sharpy.PSDs






folder = "GW_injection_example"
 
id = 0
def prior(params):
    return 0.


psd = sharpy.PSDs.__path__[0] + "/LIGO-P1200087-v18-aLIGO_DESIGN_psd.dat"



detector_settings = {
        "H1": {
            "psd_file"  : psd,

            
        },
        "L1": {
            "psd_file"  : psd, 
        },
        "V1": {
            "psd_file"  : psd,
        }
            
        



                }


from sharpy.GW_likelihood import GWNetwork, log_likelihood_det


truth =  jnp.array([3.0, 1.0, 5.5, jnp.pi/2, jnp.pi, jnp.pi/2, 30.0, 0.7, 0.0, -0.1, 0.1])
from sharpy.GW_likelihood import GWNetwork, log_likelihood_det

gw_network = GWNetwork(detector_settings,
                        injection_parameters=truth,
                        )


batched_detector = gw_network.batched_detector
log_likelihood   = partial(log_likelihood_det, detector_list=batched_detector)




prior_bounds            = jnp.array([[0., 2*jnp.pi], [-jnp.pi/2, jnp.pi/2], [4.9, 8.7], [0., jnp.pi], [0., 2*jnp.pi], [0., jnp.pi], [25, 35], [0.4, 1.], [-1e-1, 1e-1], [-1., 1.], [-1., 1.]])
boundary_conditions     = jnp.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]) #1: periodic, 0: reflective


number_of_particles     = 5000
step_size               = 0.2
alpha                   = 0.8


temperature_schedule    = jnp.concatenate((jnp.array([1e-5]),  jnp.array([1e-4]),jnp.array([1e-3]), jnp.array([5e-3]), jnp.logspace(-2, 0, 30),))

parameters_names        =  ['ra','dec','logdistance','theta_jn','phiref','pol', 'mc','q', 'tc', 'chi1', 'chi2']


folder                  = f"{folder}/run_{id}"
label                   = f"run_{id}"

if not os.path.exists(folder):
    os.makedirs(folder)

from sharpy.smc_functions import run_smc

start     = time.time()





result_dict = run_smc(log_likelihood, 
                    prior, 
                    prior_bounds,
                    boundary_conditions, 
                    alpha,
                    number_of_particles, 
                    step_size,   
                    jax.random.PRNGKey(42),
                    folder = ".",
                    label = "run",
                    initial_logZ = 0.0,
                    initial_dlogZ = 0.0,

                                            
                                                )





samples     = result_dict['posterior_samples']
logZ, dlogZ = result_dict['logZ'], result_dict['dlogZ']
                    

from corner import corner
fig = corner(np.array(samples), 
            show_titles    =True,
            truths     = truth,
            labels          = parameters_names, 
            title_kwargs   = {"fontsize": 12},)

fig.savefig(f"{folder}/{label}_corner.png")



# print(particles.shape)


sampling_time = time.time()-start
print("time = {}".format(sampling_time))


