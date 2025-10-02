
import os
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2000"

#enable jax debugging
# os.environ["JAX_LOG_COMPILES"] = "1"
import jax

# This sets the cache directory globally
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")



#uncomment these line if you want to run on cpu
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2000"

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False)  # Enable 64-bit precision
# jax.config.update("jax_enable_x64", True) 
import matplotlib.pyplot as plt
from functools import partial 
from jax import random, lax
import blackjax
from functools import partial
import time 
import json



from jax.scipy.special import logsumexp
 


def log_likelihood(params):
    dimensions = 15
    # mean1 = jnp.array([-1. for _ in range(dimensions)])
    # mean2 = jnp.array([1. for _ in range(dimensions)])
    mean1   = jnp.ones(dimensions) * -1.
    mean2   = jnp.ones(dimensions) * 1.
    
    cov = jnp.eye(dimensions) * 0.1
    inv_cov = jnp.linalg.inv(cov)
    
    diff1 = params - mean1
    diff2 = params - mean2

    exponent1 = -0.5 * jnp.einsum('...i,ij,...j->...', diff1, inv_cov, diff1)
    exponent2 = -0.5 * jnp.einsum('...i,ij,...j->...', diff2, inv_cov, diff2)
    
    norm_const = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov))
    logpdf1 = exponent1 + norm_const - jnp.log(2)  # log(0.5) = -log(2)
    logpdf2 = exponent2 + norm_const - jnp.log(2)  # log(0.5) = -log(2)
    
    # Combine using logsumexp with mixture weights
    return logsumexp(jnp.stack([logpdf1, logpdf2]), axis=0) 

# def log_likelihood(params):
#     mean = jnp.array([0.0, 0.0])
#     cov  = jnp.array([[1.10, 0.], [0., 0.10]])
#     inv_cov = jnp.linalg.inv(cov)
#     diff = params - mean
#     exponent = -0.5 * jnp.einsum('...i,ij,...j->...', diff, inv_cov, diff)
#     norm_const = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov))
#     return exponent  + norm_const


# def log_likelihood(params):
#     dimension = 50
#     mean = jnp.zeros(dimension)
#     cov  = jnp.eye(dimension)*1.
#     inv_cov = jnp.linalg.inv(cov)
#     diff = params - mean
#     exponent = -0.5 * jnp.einsum('...i,ij,...j->...', diff, inv_cov, diff)
#     norm_const = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov))
#     return exponent  + norm_const

def prior(params):
    return 0. # Uniform prior within bounds, log(1) = 0

# def log_posterior(params, beta=1):
#     return log_likelihood(params)*beta + prior(params)

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


truth =  jnp.array([3.0, 1.0, 5.5, jnp.pi/2, jnp.pi, jnp.pi/2, 30.0, 0.7, 0.0, 0.0, 0.0])
print(log_likelihood(truth))



def log_posterior(params, beta=1):
    return log_likelihood(params)*beta + prior(params)



prior_bounds =jnp.array([[0., 2*jnp.pi], [-jnp.pi/2, jnp.pi/2], [4.9, 8.7], [0., jnp.pi], [0., 2*jnp.pi], [0., jnp.pi], [25, 35], [0.4, 1.], [-1e-1, 1e-1], [-1., 1.], [-1., 1.]])

# parameters = jnp.array([4.0, 0.0, 5.5, jnp.pi/2, jnp.pi, jnp.pi/2, 30.0, 0.7, 0.0])


# print("Test log likelihood: ", log_likelihood(parameters))

# import sys
# sys.exit()





# prior_bounds            =jnp.array([[0., 2*jnp.pi], [-jnp.pi/2, jnp.pi/2], [3., 6.], [0., jnp.pi], [0., 2*jnp.pi], [0., jnp.pi], [6, 14], [0.4, 1.], [-1e-1, 1e-1]])
boundary_conditions     = jnp.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])# 0: periodic, 1: reflective


  
number_of_particles     = 6000




# temperature_schedule    = jnp.concatenate(jnp.logspace(-2, 0, 20),)
# temperature_schedule    = jnp.logspace(-3, 0, 30)
temperature_schedule    = jnp.concatenate((jnp.array([1e-4]),  jnp.array([5e-4]), jnp.array([1e-3]), jnp.array([5e-3]), jnp.logspace(-2, 0, 30),))
# temperature_schedule    = jnp.concatenate((jnp.array([1e-5]),  jnp.array([5e-5]), jnp.array([1e-4]), jnp.array([5e-3]), jnp.logspace(-2, 0, 30),))
# temperature_schedule    =jnp.logspace(-1, 0, 10)

# temperature_schedule    = temperature_schedule[1:]
parameters_names        =  ['ra','dec','logdistance','theta_jn','phiref','pol', 'mc','q', 'tc', 'chi1', 'chi2']


folder                  = "GW150914_12"
label                   = "run1 "
if not os.path.exists(folder):
    os.makedirs(folder)







from smc_functions import run_smc, run_persistent_smc


start  = time.time()




def step_size_fn(dimensions):
    return 2e-1/jnp.sqrt(dimensions)

dimensions = prior_bounds.shape[0]
# 
# step_size = 0.2
# step_size = step_size_fn(dimensions)
step_size = 0.1

particles, weights, logZ, logZerr          = run_persistent_smc(log_likelihood, 
                                                prior, 
                                                prior_bounds, 
                                                boundary_conditions, 
                                                temperature_schedule, 
                                                number_of_particles, 
                                                step_size,   
                                                master_key=jax.random.PRNGKey(1),
                                               
                                                )




from smc_functions import draw_iid_posterior_samples

samples = draw_iid_posterior_samples(particles, dimensions)
print("the number of samples after rejection sampling is:", len(samples))



# print(particles)
import numpy as np
# samples = np.array(resampled_particles[:,:len(prior_bounds)])

np.savetxt(os.path.join(folder,"posterior_samples.txt"),np.array(samples),)




from corner import corner
fig = corner(np.array(samples), 
             show_titles    =True,
            #  truths     = truth,
            labels          = parameters_names, 
             title_kwargs   = {"fontsize": 12},)

fig.savefig(f"{folder}/{label}_corner.png")



print(particles.shape)


sampling_time = time.time()-start
print("time = {}".format(sampling_time))

np.savetxt(os.path.join(folder,"sampling_time.txt"),np.array([sampling_time]))
np.savetxt(os.path.join(folder,"evidence.txt"),np.array([logZ, logZerr])[np.newaxis], fmt = '%3f')

import shutil 
shutil.copy(__file__, os.path.join(folder, "pe_script_copy" + '.py'))
