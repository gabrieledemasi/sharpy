
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



from blackjax.mcmc import integrators
kernel = blackjax.nuts.build_kernel(integrators.velocity_verlet)















 





def log_posterior(params, beta=1):
    return log_likelihood(params)*beta + prior(params)


prior_bounds = jnp.array([[-5, 5], [-5, 5]])
boundary_conditions = jnp.array([0, 0])  # 0: periodic, 1: reflective
number_of_particles= 3000
step_size = 0.1

temperature_schedule = jnp.logspace(-3, 0, 20)
# temperature_schedule = temperature_schedule[1:]




from blackjax.mcmc import integrators
kernel = blackjax.nuts.build_kernel( prior_bounds, boundary_conditions, integrators.velocity_verlet)

dimension = len(prior_bounds)


initial_position = jax.random.uniform(
    jax.random.PRNGKey(1),
    shape=(number_of_particles, dimension),
    minval=prior_bounds[:, 0],
    maxval=prior_bounds[:, 1]
)



    # return (jax.vmap(single, in_axes=(0, None)))

from smc_functions import build_mass_matrix_fn, build_kernel_fn, multinomial_resample
from smc_functions import compute_weight_and_ess_fn, make_smc_step_fn

mass_matrix_fn              = build_mass_matrix_fn(log_posterior)
kernel_fn                   = build_kernel_fn(kernel, log_posterior, step_size)
compute_weight_and_ess      = compute_weight_and_ess_fn(log_posterior)
init_fn                     = (jax.vmap(blackjax.nuts.init, in_axes=(0, None, None)))
make_a_step_vectorized      = make_smc_step_fn(init_fn, kernel_fn, log_posterior)


samples = initial_position
weights = jnp.ones(number_of_particles) / number_of_particles






def step_for(samples, beta,beta_prev, weights,  resampling_key, mutation_keys):
    print(beta, beta_prev)
    samples = multinomial_resample(resampling_key, samples, weights)
    matrices = mass_matrix_fn(samples, beta)
    jax.debug.print('matrices_computed')
    
    
    # Mutation
    samples = make_a_step_vectorized(samples, mutation_keys, beta, matrices)

    jax.debug.print("Mutation done: {samples}", samples=samples)
    # Weight update
    weights, ess = compute_weight_and_ess(samples, beta, beta_prev)

    


    return samples, weights, ess


def run_smc(initial_samples, initial_beta, temperature_schedule, initial_weights, master_key):
    n_steps = len(temperature_schedule)

    mutation_keys = random.split(master_key,(n_steps, M))
    
    resampling_keys = random.split(master_key+42, n_steps)
    

    
    beta_prev = initial_beta
    weights = initial_weights
    samples = initial_samples

    for step in range(n_steps):
        
        beta = temperature_schedule[step]
        resampling_key = resampling_keys[step]
        mutation_key = mutation_keys[step]

        samples, weights, ess = step_for(samples, beta, beta_prev,weights, resampling_key, mutation_key)
        print("ess = {}".format(ess))
        beta_prev = beta

    return samples







initial_beta = 0.0
initial_weights = jnp.ones(M) / M

start  = time.time()
final_samples= run_smc(initial_position,  initial_beta, temperature_schedule, initial_weights, jax.random.PRNGKey(i))

samples = final_samples

print("hai gia finito?")



sampling_time = time.time()-start
print("time = {}".format(sampling_time))


outdir = os.path.join(folder, label)

if not os.path.exists(outdir):
    os.makedirs(outdir)

from corner import corner
import numpy as np
np.savetxt(os.path.join(outdir,"samples.txt"),np.array(samples),)
np.savetxt(os.path.join(outdir,"truth.txt"),np.array(truth),)
np.savetxt(os.path.join(outdir,"sampling_time.txt"),np.array([sampling_time]))
np.savetxt(os.path.join(outdir, "snr.txt"),  np.array([snr]) )


fig = corner(np.array(samples), 
                truths=truth,  
                show_titles=True, title_kwargs={"fontsize": 12}, labels = names)

fig.savefig(os.path.join(outdir, "smc.png"))
    



