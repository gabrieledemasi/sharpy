import jax
from sharpy.utils import compute_mass_matrix
import jax.numpy as jnp 
from jax import random
from blackjax.mcmc import integrators
import blackjax
import numpy as np 
from netket.jax import vmap_chunked
import json
import os

def build_mass_matrix_fn(log_posterior):
    #build mass matrix function
    def single(pos, beta):
        logdensity = lambda x: log_posterior(x, beta)
        return compute_mass_matrix(logdensity, pos)
    #use vmap_chunked to avoid OOM for large number of particles
    return jax.jit(vmap_chunked(single, in_axes=(0, None),chunk_size = 1000, axis_0_is_sharded=False)) 



def mutation_step_fn(init_fn, kernel_fn,log_posterior):
    #build mutation step with NUTS kernel
    def mutation_step(position, keys, beta, matrices):

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
        
        beta_diff               = beta_after - beta_before
        log_weights             = jax.vmap(log_likelihood,)(samples) * beta_diff
        stabilized_log_weights  = log_weights - jnp.max(log_weights)
        log_ess                 = 2 * jax.scipy.special.logsumexp(stabilized_log_weights) - jax.scipy.special.logsumexp(2 * stabilized_log_weights)
        ess                     = jnp.exp(log_ess)
        log_weights             = log_weights.flatten()

        return log_weights, ess
    
    return compute_weight_and_ess


def smc_step_fn(mass_matrix_fn, mutation_step_vectorized, compute_weight_and_ess):
    
    # Single SMC step
    def smc_step(samples, beta,beta_prev, weights,  resampling_key, mutation_keys):

        log_weights, ess                = compute_weight_and_ess(samples, beta, beta_prev)
        weights                         = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))
        index                           = jax.random.choice(resampling_key, np.arange(len(samples)), (len(samples),), p=weights)
        samples                         = samples[index]
        # Mutation
        matrices                        = mass_matrix_fn(samples, beta)
        samples                         = mutation_step_vectorized(samples, mutation_keys, beta, matrices)

        return samples, log_weights, ess
    
    return smc_step







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

        log_evidence_piece = logsumexp(result[key]['log_weights']) - np.log(len(result[key]['log_weights']))
            
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
    M           = np.max( log_likelihoods - log_posterior_primed)  
    u           = np.random.uniform( size = len(log_posterior_primed))
    accepted    =  +log_likelihoods - log_posterior_primed - M > np.log(u)
    samples     = samples[accepted]


    return samples



def compute_evidence(result_dict):

    
    log_evidence = 0.0
    errors       = []
    
    log_weight_list = [result_dict[key]['log_weights'] for key in result_dict.keys()]
    print(log_weight_list[0][0])
    

    for key in result_dict.keys():
        
        log_evidence_piece = logsumexp(result_dict[key]['log_weights']) - np.log(len(result_dict[key]['log_weights']))

        ess = result_dict[key]['ess']
        
        
        log_evidence      += log_evidence_piece

        ### compute evidence with bootstraping
        # log_boot_weights   = jnp.array(result_dict[key]['log_weights'])
        # dlogz_piece        = np.var([logsumexp(log_boot_weights[np.random.choice(len(log_boot_weights), len(log_boot_weights))])  for _ in range(100)])
        
        # errors.append(dlogz_piece)#*(1+len(result_dict[key]['log_weights'])/ess))

        #delta methods
        weights = jnp.exp(result_dict[key]['log_weights'].copy()-np.max(result_dict[key]['log_weights']))
        dlogz_piece       = np.var(weights) /((np.mean(weights))**2*len(weights))*(1 +ess/len(weights))
        errors.append(dlogz_piece)

    
    
    logz, dlogz  = log_evidence, np.sqrt(np.sum((errors)))
  

    return logz, dlogz










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
    


    if not os.path.exists(folder):
        os.makedirs(folder)



    #Define the log-posterior
    def log_posterior(params, beta=1):
        return log_likelihood(params)*beta + prior(params)

    #Set up the SMC components
    kernel                      = blackjax.nuts.build_kernel( prior_bounds, boundary_conditions, integrators.velocity_verlet)
    mass_matrix_fn              = build_mass_matrix_fn(log_posterior)
    kernel_fn                   = build_kernel_fn(kernel, log_posterior, step_size)
    compute_weight_and_ess      = compute_weight_and_ess_fn(log_likelihood)
    init_fn                     = (jax.vmap(blackjax.nuts.init, in_axes=(0, None, )))
    mutation_step_vectorized    = mutation_step_fn(init_fn, kernel_fn, log_posterior)
    step_for                    = smc_step_fn(mass_matrix_fn, mutation_step_vectorized, compute_weight_and_ess, )
    vmapped_likelihood          = jax.jit(jax.vmap(log_likelihood))
    smc_dict                   = {}        

    #Generate initial particles from the prior
    initial_position = jax.random.uniform(
                                        jax.random.PRNGKey(1),
                                        shape=(number_of_particles, len(prior_bounds)),
                                        minval=prior_bounds[:, 0],
                                        maxval=prior_bounds[:, 1]
                                        )
    
    
    #initialize SMC
    n_steps         = len(temperature_schedule)
    mutation_keys   = random.split(master_key,(n_steps, number_of_particles))
    resampling_keys = random.split(master_key+42, n_steps)
    initial_beta    = 0.0
    initial_weights = jnp.ones(number_of_particles) / number_of_particles
    beta_prev       = initial_beta
    weights         = initial_weights
    samples         = initial_position
    
    #SMC main loop
    for step in range(n_steps):

        smc_dict[int(step)] = {}

        beta                    = temperature_schedule[step]
        resampling_key          = resampling_keys[step]
        mutation_key            = mutation_keys[step]

        #Do a SMC step
        samples, log_weights, ess   = step_for(samples, beta, beta_prev,weights, resampling_key, mutation_key)
        print("ess = {}".format(ess))
        if jnp.isnan(ess):
            print("ESS is NaN, stopping SMC.")
            break

        #Store SMC step results
        smc_dict[step]["samples"]           = np.array(samples).tolist()
        smc_dict[step]["log_weights"]       = np.array(log_weights).tolist()
        smc_dict[step]["ess"]               = float(ess)
        smc_dict[step]['log_likelihoods']   = np.array(vmapped_likelihood(samples)).tolist()
        smc_dict[step]['beta']              = float(beta)
        beta_prev                           = beta


    #compute evidence and draw iid samples using rejection sampling
    posterior_samples       = draw_iid_samples(smc_dict)
    logZ, dlogZ             = compute_evidence(smc_dict)

    #save results
    result_dict = {}
    result_dict['SMC']      = smc_dict
    result_dict['logZ']     = float(logZ)
    result_dict["dlogZ"]    = float(dlogZ)
    result_dict['posterior_samples'] = posterior_samples.tolist()

    # with open(f"{folder}/{label}_result.json", "w") as f:
    #     json.dump(result_dict, f)
    
    return result_dict




        






