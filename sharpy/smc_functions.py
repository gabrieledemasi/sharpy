import jax
from sharpy.utils import compute_mass_matrix
import jax.numpy as jnp 
from jax import random
from blackjax.mcmc import integrators
import blackjax

import numpy as np 
from netket.jax import vmap_chunked
import json


def build_mass_matrix_fn(log_posterior):
    def single(pos, beta):
        logdensity = lambda x: log_posterior(x, beta)
        
        return compute_mass_matrix(logdensity, pos)
    return jax.jit(vmap_chunked(single, in_axes=(0, None),chunk_size = 1000, axis_0_is_sharded=False)) 



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
    batched_kernel = jax.jit(vmap_chunked(_kernel, in_axes=(0, 0, 0, 0), chunk_size = 9000, axis_0_is_sharded=False))
    return batched_kernel




@jax.jit
def multinomial_resample(key, particles, weights, ):
    cdf = jnp.cumsum(weights)
    u = jax.random.uniform(key, shape=(len(weights),))
    idx = jnp.searchsorted(cdf, u)
    return particles[idx]


def multinomial_resample_fn(number_of_particles):
    @jax.jit
    def multinomial_resample(key, particles, weights, ):
        cdf = jnp.cumsum(weights)
        u   = jax.random.uniform(key, shape=(number_of_particles,))
        idx = jnp.searchsorted(cdf, u)
        return particles[idx]
    return multinomial_resample


def compute_weight_and_ess_fn(log_likelihood):
    @jax.jit
    def compute_weight_and_ess(samples, beta_after, beta_before):
        

        beta_diff = beta_after - beta_before
        log_weights = jax.vmap(log_likelihood,)(samples) * beta_diff
        
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

  
        samples                         = jax.random.choice(resampling_key, samples, (len(samples),), p=weights)
        # Mutation
        matrices                        = mass_matrix_fn(samples, beta)
        samples                         = mutation_step_vectorized(samples, mutation_keys, beta, matrices)

        return samples, weights, weights_nonorm, ess
    
    
    
    return smc_step











def run_smc(log_likelihood, 
            prior, 
            prior_bounds,
            boundary_conditions, 
            temperature_schedule, 
            number_of_particles, 
            step_size,   
            master_key,
            folder = ".",
            label = "run",
            ):

    def log_posterior(params, beta=1):
        return log_likelihood(params)*beta + prior(params)

    kernel                      = blackjax.nuts.build_kernel( prior_bounds, boundary_conditions, integrators.velocity_verlet)
    
    mass_matrix_fn              = build_mass_matrix_fn(log_posterior)
    kernel_fn                   = build_kernel_fn(kernel, log_posterior, step_size)
    compute_weight_and_ess      = compute_weight_and_ess_fn(log_likelihood)
    init_fn                     = (jax.vmap(blackjax.nuts.init, in_axes=(0, None, )))
    mutation_step_vectorized    = mutation_step_fn(init_fn, kernel_fn, log_posterior)
    step_for                    = smc_step_fn(mass_matrix_fn, mutation_step_vectorized, compute_weight_and_ess, )
    vmapped_likelihood          = jax.jit(jax.vmap(log_likelihood))
    smc_dict                   = {}        

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

        smc_dict[int(step)] = {}

        beta                    = temperature_schedule[step]
        resampling_key          = resampling_keys[step]
        mutation_key            = mutation_keys[step]

        samples, weights_nonorm, weights, ess   = step_for(samples, beta, beta_prev,weights, resampling_key, mutation_key)

        if jnp.isnan(ess):
            print("ESS is NaN, stopping SMC.")
            break

        smc_dict[step]["samples"]           = np.array(samples).tolist()
        smc_dict[step]["weights"]           = np.array(weights).tolist()
        smc_dict[step]["ess"]               = float(ess)
        smc_dict[step]['log_likelihoods']   = np.array(vmapped_likelihood(samples)).tolist()
        smc_dict[step]['beta']              = float(beta)


        print("ess = {}".format(ess))
        
        beta_prev                               = beta

    
    posterior_samples                 = draw_iid_samples(smc_dict)
    logZ, dlogZ             = compute_evidence(smc_dict)


    result_dict = {}
    result_dict['SMC']      = smc_dict
    result_dict['logZ']     = float(logZ)
    result_dict["dlogZ"]    = float(dlogZ)
    result_dict['posterior_samples'] = posterior_samples.tolist()

    with open(f"{folder}/{label}_result.json", "w") as f:
        json.dump(result_dict, f)
    
    return result_dict




        
from scipy.special import logsumexp


def draw_iid_samples(dict):
    result = dict 
    samples         = []
    log_likelihoods = []
    betas           = []
    log_evidences   = []
    log_evidence    = 0.0 #this is the evidence of the prior 

    for key in result.keys():
        

        samples         += list(result[key]['samples'])
        log_likelihoods += list((result[key]['log_likelihoods']))
        betas.append(result[key]['beta'])

        log_evidence_piece = logsumexp(np.log(result[key]['weights'])) - np.log(len(result[key]['weights']))
            
        log_evidence      += log_evidence_piece
        log_evidences.append(log_evidence)
        

   
    betas                   = np.array(betas)
    log_evidences           = np.array(log_evidences)
    samples                 = np.array(samples)
    log_likelihoods         = np.array(log_likelihoods)

    "construct mixture posterior"
    log_posterior_primed        = np.array([log_likelihoods * beta - log_evidence for beta, log_evidence in zip(betas, log_evidences)])
    log_posterior_primed        = jnp.logaddexp.reduce( log_posterior_primed, axis = 0) - jnp.log(len(result.keys()))

 
    #rejection sampling
    M = np.max( log_likelihoods - log_posterior_primed)  
    u = np.random.uniform( size = len(log_posterior_primed))
    accepted =  +log_likelihoods - log_posterior_primed - M > np.log(u)
    samples = samples[accepted]


    return samples
    





def compute_log_z_piece(key, log_weights):
        indices             = jax.random.choice(key, len(log_weights), (len(log_weights),))
        log_weights_boot    = log_weights[indices]
        log_z_piece         = jnp.logaddexp.reduce(log_weights_boot) - jnp.log(len(log_weights_boot))
        return log_z_piece

def compute_bootstrap_variance(key, log_weights):    
    keys        = jax.random.split(key, 100)
    log_zs      = jax.vmap(compute_log_z_piece,in_axes = (0, None)) (keys, log_weights)
    variance    = jnp.var(log_zs)
    return variance
        

def compute_log_weights_and_log_z(likelihoods, beta, evidence, current_beta):

    log_numerator           = likelihoods * current_beta
    log_denominator         = likelihoods * beta[:, None] - evidence[:, None]
    log_denominator         = jnp.logaddexp.reduce( log_denominator, axis = 0) - jnp.log(len(beta))
    log_weights             = log_numerator - log_denominator
    log_z                   = jnp.logaddexp.reduce(log_weights) - np.log(len(log_weights))

    return log_weights, log_z



def compute_evidence(result_dict):

    
    log_evidence = 0.0
    errors       = []
    

    for key in result_dict.keys():
        
        log_evidence_piece = logsumexp(np.log(result_dict[key]['weights'])) - np.log(len(result_dict[key]['weights']))
        
        log_evidence      += log_evidence_piece

        ### compute evidence with bootstraping
        log_boot_weights   = np.log(result_dict[key]['weights'])
        dlogz_piece        = np.var([logsumexp(log_boot_weights[np.random.choice(len(log_boot_weights), len(log_boot_weights))]) - len(log_boot_weights) for _ in range(100)])

        errors.append(dlogz_piece)
        
    
    logz, dlogz  = log_evidence, np.sqrt(np.sum(np.cumsum(errors)))

    return logz, dlogz








