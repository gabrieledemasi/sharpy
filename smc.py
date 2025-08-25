
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



from blackjax.mcmc import integrators
kernel = blackjax.nuts.build_kernel(integrators.velocity_verlet)















 





def log_posterior(params, beta=1):
    return log_likelihood(params)*beta + prior(params)


prior_bounds = jnp.array([[-5, 5], [-5, 5]])
boundary_conditions = jnp.array([0, 0])  # 0: periodic, 1: reflective
number_of_particles= 3000


from blackjax.mcmc import integrators
kernel = blackjax.nuts.build_kernel( prior_bounds, boundary_conditions, integrators.velocity_verlet)

dimension = len(prior_bounds)


initial_position = jax.random.uniform(
    jax.random.PRNGKey(1),
    shape=(number_of_particles, dimension),
    minval=prior_bounds[:, 0],
    maxval=prior_bounds[:, 1]
)


def build_mass_matrix_fn():
    def single(pos, beta):
        logdensity = lambda x: log_posterior(x, beta)
        
        return compute_mass_matrix(logdensity, pos)
    return jax.jit(jax.vmap(single, in_axes=(0, None)))
    # return (jax.vmap(single, in_axes=(0, None)))


mass_matrix_fn = build_mass_matrix_fn()



















temperature_schedule = jnp.logspace(-3, 0, 20)
temperature_schedule = temperature_schedule[1:]

samples = initial_position
weights = jnp.ones(M) / M

step_size = 0.1
from functools import partial





print(weights, initial_position)




import time 



init_fn = (jax.vmap(blackjax.nuts.init, in_axes=(0, None, None)))



def build_kernel_fn(step_size):
    def _kernel(rng_key, state, beta, metric):
        logdensity_fn = lambda x: log_posterior(x, beta)
        return kernel(rng_key, state, logdensity_fn, step_size, metric)

    # JIT-compile the batched kernel function
    batched_kernel = jax.jit(jax.vmap(_kernel, in_axes=(0, 0, 0, 0), out_axes=(0, 0)))
    return batched_kernel


kernel_fn =build_kernel_fn(step_size)






def make_a_step_vectorized(position, keys, beta ,matrices):
    """
    position: (M, D)
    beta: scalar or vector
    """
    logdensity_fn = lambda x: log_posterior(x, beta)  # Only for init, not passed into JIT

    # Initialize state
    state = init_fn(position, logdensity_fn, step_size)

    beta_batch = jnp.broadcast_to(beta, (position.shape[0],))

    state, info = kernel_fn(keys, state,  beta_batch, matrices)

    return state.position



@jax.jit
def compute_weight_and_ess(samples, beta_after, beta_before):
    def log_density_diff(x, beta):
        return log_posterior(x, beta)

    beta = beta_after - beta_before
    log_weights = jax.vmap(log_density_diff, in_axes=(0, None))(samples, beta)
    
    log_weights = log_weights - jnp.max(log_weights)
    weights = jnp.exp(log_weights)
    weights = weights / jnp.sum(weights)
    ess = (jnp.sum(weights)) ** 2 / jnp.sum(weights**2)

    return weights, ess

@jax.jit
def multinomial_resample(key, particles, weights):
    cdf = jnp.cumsum(weights)
    u = jax.random.uniform(key, shape=(len(weights),))
    idx = jnp.searchsorted(cdf, u)
    return particles[idx]





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
    



