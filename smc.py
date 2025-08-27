
import os

#enable jax debugging
# os.environ["JAX_LOG_COMPILES"] = "1"


#uncomment these line if you want to run on cpu
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2000"

import jax
import jax.numpy as jnp
# jax.config.update("jax_enable_x64", False)  # Enable 64-bit precision

import matplotlib.pyplot as plt
from functools import partial 
from jax import random, lax
import blackjax
from functools import partial
import time 


















 
def prior(params):
    return  0.0  # Uniform prior within bounds, log(1) = 0

def log_likelihood(params):
    mean = jnp.array([0.0, 0.0])
    cov  = jnp.array([[1.0, 0.], [0., 1.0]])
    inv_cov = jnp.linalg.inv(cov)
    diff = params - mean
    exponent = -0.5 * jnp.einsum('...i,ij,...j->...', diff, inv_cov, diff)
    norm_const = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov))
    return exponent + norm_const




def log_posterior(params, beta=1):
    return log_likelihood(params)*beta + prior(params)


prior_bounds            = jnp.array([[-5, 5], [-5, 5]])
boundary_conditions     = jnp.array([0, 0])  # 0: periodic, 1: reflective
number_of_particles     = 2000
step_size               = 0.1
temperature_schedule    = jnp.logspace(-2, 0, 10)
temperature_schedule    = temperature_schedule[1:]
parameters_names        = None
truth                   = None

folder                  = "results"
label                   = "smc_2d_gaussian"



from smc_functions import run_smc




start  = time.time()


final_samples = run_smc(log_posterior, 
                        prior_bounds, 
                        boundary_conditions, 
                        temperature_schedule, 
                        number_of_particles, 
                        step_size,   
                        master_key=jax.random.PRNGKey(0)
                        )

samples = final_samples

sampling_time = time.time()-start
print("time = {}".format(sampling_time))



####PLOTTING###

outdir = os.path.join(folder, label)

if not os.path.exists(outdir):
    os.makedirs(outdir)

from corner import corner
import numpy as np
np.savetxt(os.path.join(outdir,"samples.txt"),np.array(samples),)

np.savetxt(os.path.join(outdir,"sampling_time.txt"),np.array([sampling_time]))


if truth is not None:
    np.savetxt(os.path.join(outdir,"truth.txt"),np.array(truth),)

# np.savetxt(os.path.join(outdir, "snr.txt"),  np.array([snr]) )



# 

fig = corner(np.array(samples), 
                truths=truth,  
                show_titles=True, 
                title_kwargs={"fontsize": 12}, 
                labels = parameters_names)


fig.savefig(os.path.join(outdir, f"{label}_corner.png"))
    



