import jax
from utils import compute_mass_matrix
import jax.numpy as jnp 
from jax import random
from blackjax.mcmc import integrators
import blackjax

import numpy as np 





def build_mass_matrix_fn(log_posterior):
    def single(pos, beta):
        logdensity = lambda x: log_posterior(x, beta)
        
        return compute_mass_matrix(logdensity, pos)
    return jax.jit(jax.vmap(single, in_axes=(0, None)))



def mutation_step_fn(init_fn, kernel_fn,log_posterior):

    def mutation_step(position, keys, beta, matrices):
        """
        position: (M, D)
        beta: scalar or vector
        """
        logdensity_fn   = lambda x: log_posterior(x, beta)  # Only for init, not passed into JIT

        # Initialize state
        state           = init_fn(position, logdensity_fn,)
        

        beta_batch      = jnp.broadcast_to(beta, (position.shape[0],))
        state, _        = kernel_fn(keys, state,  beta_batch, matrices)

        return state.position
    
    return mutation_step


def build_kernel_fn(kernel, log_posterior, step_size):
    def _kernel(rng_key, state, beta, metric):
        logdensity_fn = lambda x: log_posterior(x, beta)

        return kernel(rng_key, state, logdensity_fn, step_size, metric)

    # JIT-compile the batched kernel function
    batched_kernel = jax.jit(jax.vmap(_kernel, in_axes=(0, 0, 0, 0), out_axes=(0, 0)))
    return batched_kernel





@jax.jit
def multinomial_resample(key, particles, weights):
    cdf = jnp.cumsum(weights)
    u = jax.random.uniform(key, shape=(len(weights),))
    idx = jnp.searchsorted(cdf, u)
    return particles[idx]



def compute_weight_and_ess_fn(log_posterior):
    @jax.jit
    def compute_weight_and_ess(samples, beta_after, beta_before):
        def log_density_diff(x, beta):
            return log_posterior(x, beta)

        beta = beta_after - beta_before
        log_weights = jax.vmap(log_density_diff, in_axes=(0, None))(samples, beta)
        
        log_weights = log_weights #- jnp.max(log_weights)
        weights_nonorm = jnp.exp(log_weights)
        weights = weights_nonorm / jnp.sum(weights_nonorm)
        ess = (jnp.sum(weights)) ** 2 / jnp.sum(weights**2)

        return weights,weights_nonorm, ess
    return compute_weight_and_ess


def smc_step_fn(mass_matrix_fn, mutation_step_vectorized, compute_weight_and_ess):

    def smc_step(samples, beta,beta_prev, weights,  resampling_key, mutation_keys):


        # jax.debug.print("Mutation done: {samples}", samples=samples)
        # Reweighting
        weights, weights_nonorm,  ess    = compute_weight_and_ess(samples, beta, beta_prev)

        # Resampling
        samples         = multinomial_resample(resampling_key, samples, weights)
        # Mutation
        matrices        = mass_matrix_fn(samples, beta)
        
        samples         = mutation_step_vectorized(samples, mutation_keys, beta, matrices)

        return samples, weights, weights_nonorm, ess
    
    
    
    return smc_step











def run_smc(log_posterior, prior_bounds, boundary_conditions, temperature_schedule, number_of_particles, step_size,   master_key):

    kernel                 = blackjax.nuts.build_kernel( prior_bounds, boundary_conditions, integrators.velocity_verlet)

    mass_matrix_fn              = build_mass_matrix_fn(log_posterior)
    kernel_fn                   = build_kernel_fn(kernel, log_posterior, step_size)
    compute_weight_and_ess      = compute_weight_and_ess_fn(log_posterior)
    init_fn                     = (jax.vmap(blackjax.nuts.init, in_axes=(0, None, )))
    mutation_step_vectorized    = mutation_step_fn(init_fn, kernel_fn, log_posterior)
    step_for                    = smc_step_fn(mass_matrix_fn, mutation_step_vectorized, compute_weight_and_ess, )

    samples_dict                 = {}        

    initial_position = jax.random.uniform(
                                        jax.random.PRNGKey(1),
                                        shape=(number_of_particles, len(prior_bounds)),
                                        minval=prior_bounds[:, 0],
                                        maxval=prior_bounds[:, 1]
                                        )
    
    

    n_steps         = len(temperature_schedule)

    mutation_keys   = random.split(master_key,(n_steps, number_of_particles))
    resampling_keys = random.split(master_key+42, n_steps)

    initial_beta    = 0.0
    initial_weights = jnp.ones(number_of_particles) / number_of_particles

    
    beta_prev       = initial_beta
    weights         = initial_weights
    samples         = initial_position

    for step in range(n_steps):

        samples_dict[int(step)] = {}

        beta                    = temperature_schedule[step]
        resampling_key          = resampling_keys[step]
        mutation_key            = mutation_keys[step]

        samples, weights_nonorm, weights, ess   = step_for(samples, beta, beta_prev,weights, resampling_key, mutation_key)
        samples_dict[step]["samples"]           = np.array(samples).tolist()
        samples_dict[step]["weights"]           = np.array(weights).tolist()
        samples_dict[step]["ess"]               = float(ess)

        print("ess = {}".format(ess))
        
        beta_prev                               = beta

    return samples, samples_dict


def compute_evidence(folder, label):
    import json
    result_path = f'{folder}/{label}/result.json'
    with open(result_path, 'r') as f:
        result = json.load(f)
    evidence = 1.0
    errors    = []
    

    for key in result.keys():
        if float(key) > -1:
            
            evidence_piece = np.sum(result[key]['weights'])/len(result[key]['weights'])
            
            
            evidence      *= evidence_piece

            ### compute evidence with bootstraping
            boot_weights = np.array(result[key]['weights'])
            evidences = [np.sum(boot_weights[np.random.choice(len(boot_weights), len(boot_weights))])/len(boot_weights) for _ in range(1000)]

            dlogz_piece = np.var((np.log([np.sum(boot_weights[np.random.choice(len(boot_weights), len(boot_weights))])/len(boot_weights) for _ in range(1000)])))
            errors.append(dlogz_piece)
            
    
    logz, dlogz  = np.log(evidence), np.sqrt(np.sum(np.cumsum(errors)))
    np.savetxt(f'{folder}/{label}/evidence.txt', np.array([logz, dlogz])[np.newaxis], fmt='%3f')

    return logz, dlogz







def find_global_minimum(log_posterior, prior_bounds, boundary_conditions,  number_of_samples_single_chain, number_of_parallel_chains,  step_size,  ):


    def evolve_point(position, log_posterior, prior_bounds, boundary_conditions,  number_of_samples, step_size,   master_key):
        kernel                      = blackjax.nuts.build_kernel( prior_bounds, boundary_conditions, integrators.velocity_verlet)
        init_fn                     = (blackjax.nuts.init)

        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state, log_posterior, step_size, compute_mass_matrix(log_posterior, state.position), max_num_doublings= 6 )
            return state, state
        
        initial_state               = init_fn(position, log_posterior,)
        keys                        = jax.random.split(master_key, int(number_of_samples))
        _, states                   = jax.lax.scan(one_step, initial_state, keys)
        return states.position



    points      = jax.random.uniform(
                                    jax.random.PRNGKey(1),
                                    shape=(number_of_parallel_chains, len(prior_bounds)),
                                    minval=prior_bounds[:, 0],
                                    maxval=prior_bounds[:, 1]
                                    )
    

    keys = jax.random.split(jax.random.PRNGKey(0), number_of_parallel_chains)
    evolve_points = jax.vmap(evolve_point, in_axes=(0, None, None, None, None, None, 0))
    
    chains = evolve_points(points ,
                            log_posterior,
                            prior_bounds,
                            boundary_conditions,
                            number_of_samples_single_chain, 
                            step_size,
                            keys
                            )
    samples = np.array(chains).reshape(-1, len(prior_bounds))
    LL_values = jax.vmap(log_posterior)(jnp.array(samples))
    max_likelihood_point = samples[np.argmax(LL_values)]

    

    return  samples, max_likelihood_point



    


