
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
from sharpy.smc_functions import run_smc





   
def prior(params):
    return 0.









prior_bounds            = jnp.array([[-5, 5] for _ in range(10)])
boundary_conditions     = jnp.array([0 for _ in range(10)])
number_of_particles     = 9000
step_size               = 0.2
temperature_schedule    = jnp.concatenate((jnp.array([1e-5]),  jnp.array([1e-4]),jnp.array([1e-3]), jnp.array([5e-3]), jnp.logspace(-2, 0, 30),))
folder                  = f"Gaussian_mixture_example"
label                   = f"sharpy_run"




start     = time.time()


#Define the Gaussian Mixture Log-Likelihood
from sharpy.test_distributions import bimodal_gaussian_mixture
log_likelihood = bimodal_gaussian_mixture(mean_1=-1., mean_2=1., sigma=0.1, weight=0.5, dimensions=10)



#Run Sharpy
result_dict     = run_smc(  log_likelihood,
                            prior,
                            prior_bounds,
                            boundary_conditions,
                            temperature_schedule,
                            number_of_particles,
                            step_size,
                            master_key=jax.random.PRNGKey(jnp.array(0)),
                            folder = folder,
                                                )



#SOME POST-PROCESSING OF THE RESULTS

samples     = result_dict['posterior_samples']
print("the total number of samples is:", len(samples))
logZ, dlogZ = result_dict['logZ'], result_dict['dlogZ']
print("logZ = {}, dlogZ = {}".format(logZ, dlogZ))
                    
from corner import corner
fig = corner(np.array(samples),)
fig.savefig(f"{folder}/{label}_corner.png")

sampling_time = time.time()-start
print("The total sampling time is  = {}".format(sampling_time))





