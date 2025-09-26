
import os
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2000"

# enable jax debugging
os.environ["JAX_LOG_COMPILES"] = "1"


#uncomment these line if you want to run on cpu
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2000"

import jax
import jax.numpy as jnp
# jax.config.update("jax_enable_x64", False)  # Enable 64-bit precision
jax.config.update("jax_enable_x64", True) 
import matplotlib.pyplot as plt
from functools import partial 
from jax import random, lax
import blackjax
from functools import partial
import time 
import json



from jax.scipy.special import logsumexp
 


# def log_likelihood(params):
#     mean1 = jnp.array([-1.0, -1.0])
#     mean2 = jnp.array([1.0, 1.0])
    
#     cov  = jnp.array([[0.01, 0.], [0., 0.01]])

#     inv_cov = jnp.linalg.inv(cov)
    
#     diff1 = params - mean1
#     diff2 = params - mean2
#     exponent1 = -0.5 * jnp.einsum('...i,ij,...j->...', diff1, inv_cov, diff1)
#     exponent2 = -0.5 * jnp.einsum('...i,ij,...j->...', diff2, inv_cov, diff2)
#     norm_const = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov))
#     logpdf1 = exponent1 + norm_const
#     logpdf2 = exponent2 + norm_const
#     return logsumexp(jnp.array([logpdf1, logpdf2]))

def log_likelihood(params):
    mean = jnp.array([0.0, 0.0])
    cov  = jnp.array([[1.10, 0.], [0., 0.10]])
    inv_cov = jnp.linalg.inv(cov)
    diff = params - mean
    exponent = -0.5 * jnp.einsum('...i,ij,...j->...', diff, inv_cov, diff)
    norm_const = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov))
    return exponent  + norm_const

def prior(params):
    return 0. # Uniform prior within bounds, log(1) = 0

# def log_posterior(params, beta=1):
#     return log_likelihood(params)*beta + prior(params)

psd_file = "/leonardo/home/userexternal/gdemasi0/SHARPy-GW/LIGO-P1200087-v18-aLIGO_DESIGN_psd.dat"

detector_settings = {
        "H1": {
            "psd_file": psd_file,
            "data_file": None,
            "channel": None,
        },
        "L1": {
            "psd_file": psd_file,
            "data_file": None,
            "channel": None,
        },
        "V1": {
            "psd_file": psd_file,
            "data_file": None,
            "channel": None,
        },
    }


from likelihood import GWNetwork, log_likelihood_det


truth =  jnp.array([3.0, 0.10, 7.5, jnp.pi/3, jnp.pi/2, jnp.pi/2, 30.0, 0.7, 0.0])

from likelihood import GWNetwork, log_likelihood_det
gw_network = GWNetwork(detector_settings,
                       injection_parameters=truth,
    
                       )

batched_detector = gw_network.batched_detector

log_likelihood = partial(log_likelihood_det, detector_list=batched_detector)


def log_posterior(params, beta=1):
    return log_likelihood(params)*beta + prior(params)



prior_bounds =jnp.array([[0., 2*jnp.pi], [-jnp.pi/2, jnp.pi/2], [4.9, 8.7], [0., jnp.pi], [0., 2*jnp.pi], [0., jnp.pi], [25, 35], [0.4, 1.], [-1e-1, 1e-1]])






# prior_bounds            =jnp.array([[0., 2*jnp.pi], [-jnp.pi/2, jnp.pi/2], [3., 6.], [0., jnp.pi], [0., 2*jnp.pi], [0., jnp.pi], [6, 14], [0.4, 1.], [-1e-1, 1e-1]])
boundary_conditions     = jnp.array([1, 0, 0, 0, 1, 1, 0, 0, 0])# 0: reflective, 1: circular


number_of_particles     = 2000
step_size               = 2e-1
temperature_schedule    = jnp.logspace(-3, 0, 30)

parameters_names        = None


folder                  = "results"
label                   = "smc_2d_gaussian"


# #####Maximum Likelihood finder #####

# from smc_functions import find_global_minimum_nuts, find_global_minimum_jaxopt
# number_of_samples_single_chain = 100
# number_of_parallel_chains      = 10
# step_size                     = 1e-1 * 3

# samples, max_likelihood_point = find_global_minimum_nuts(log_posterior, prior_bounds, boundary_conditions,  number_of_samples_single_chain, 
#                                                 number_of_parallel_chains,  step_size, )


# print(max_likelihood_point)


# from likelihood import project_waveform
# projecting_map = jax.vmap(project_waveform, in_axes=(None, 0))
# waveform_truth = projecting_map(truth, batched_detector)
# waveform_max   = projecting_map(max_likelihood_point, batched_detector)


# import matplotlib.pyplot as plt
# import numpy as np
# plt.plot(np.fft.fft(waveform_truth[0]), label = "truth")
# plt.plot(np.fft.fft(waveform_max[0]),   label = "maxL")
# plt.legend()
# plt.savefig("waveform.png")
# plt.show()




# fig = corner(samples, truths = truth)
# fig.savefig("test.png")
# import sys 
# sys.exit()




from smc_functions import run_smc




start  = time.time()


final_samples, samples_dict = run_smc(log_likelihood, 
                                        prior, 


                                        prior_bounds, 
                                        boundary_conditions, 
                                        temperature_schedule, 
                                        number_of_particles, 
                                        step_size,   
                                        master_key=jax.random.PRNGKey(1)
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

# np.savetxt(os.path.join(outdir,"sampling_time.txt"),np.array([sampling_time]))
# with open(f'{folder}/{label}/result.json', 'w') as fp:
#     json.dump(samples_dict, fp)

# from smc_functions import compute_evidence
# z, dz = compute_evidence(folder, label)

# print(f"log_evidnence  = {z} +- {dz}")

# if truth is not None:
#     np.savetxt(os.path.join(outdir,"truth.txt"),np.array(truth),)

# np.savetxt(os.path.join(outdir, "snr.txt"),  np.array([snr]) )



# 

fig = corner(np.array(samples), 
                truths=truth,  
                show_titles=True, 
                title_kwargs={"fontsize": 12}, 
                labels = parameters_names)


fig.savefig(os.path.join(outdir, f"{label}_corner.png"))
    



