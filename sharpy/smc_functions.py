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
import sys


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







def find_next_beta(compute_weight_and_ess, samples, beta_prev, ess_target):
    
    beta_lower = beta_prev
    beta_upper = 1.0
    while True:
        if beta_upper - beta_lower < 1e-8:
            beta_next = beta_upper
            break
        beta_next = (beta_lower + beta_upper) / 2.0
        # print("Searching for next beta between {} and {}, current guess: {}".format(beta_lower, beta_upper, beta_next))
        ess_diff = compute_weight_and_ess(samples, beta_upper, beta_prev)[1] - ess_target
        if ess_diff > 0:
            beta_lower = beta_next
        else:
            beta_upper= beta_next
    return beta_next




# def compute_logZ(log_weights, ess , previous_logZ=0.0, previous_dlogZ=0.0,):
#     logZ               = jax.scipy.special.logsumexp(log_weights) - jnp.log(len(log_weights))

#     # dlogz_piece        = np.var([logsumexp(log_weights[np.random.choice(len(log_weights), len(log_weights))]) - len(log_weights) for _ in range(1000)])
#     epsilon            = ess/len(log_weights)
#     dlogz_piece        = np.sqrt((1.0 - epsilon) / (epsilon * len(log_weights)))
#     print("dlogz_piece =", dlogz_piece)

#     st                 = np.sum((jnp.exp(log_weights) - jnp.exp(logZ))**2)/(len(log_weights) -1)
#     var_logZ           = st / (len(log_weights)) / jnp.exp( logZ)**2
#     # dlogZ              = np.sqrt(previous_dlogZ**2 + np.log(len(log_weights))*dlogz_piece**2)
#     dlogZ             = np.sqrt(previous_dlogZ**2 + var_logZ )
#     logZ              += previous_logZ





    # return logZ, dlogZ



def run_smc(log_likelihood, 
            prior, 
            prior_bounds,
            boundary_conditions, 
            alpha,
            number_of_particles, 
            step_size,   
            master_key,
            folder = ".",
            label = "run",
            initial_particles = "prior",
            initial_logZ = 0.0,
            initial_dlogZ = 0.0
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
    smc_dict                    = {}        

    #Generate initial particles from the prior
    if initial_particles == "prior":
        initial_position = jax.random.uniform(
                                            jax.random.PRNGKey(1),
                                            shape=(number_of_particles, len(prior_bounds)),
                                            minval=prior_bounds[:, 0],
                                            maxval=prior_bounds[:, 1]
                                            )
    else:
        initial_position = initial_particles
        
    
    
    #initialize SMC
    

    initial_beta    = 0.0
    initial_weights = jnp.ones(number_of_particles) / number_of_particles
    beta_prev       = initial_beta
    weights         = initial_weights
    samples         = initial_position
    beta_next       = initial_beta
    step            = 0

    previous_logZ    = initial_logZ
    previous_dlogZ   = initial_dlogZ

    
    #SMC main loop
    while beta_next < 1.0 - 1e-8:
        beta_next = find_next_beta(compute_weight_and_ess, samples, beta_prev,ess_target= int(number_of_particles * alpha))
        # sys.exit()
        smc_dict[int(step)] = {}


        resampling_key          = random.split(master_key+42 + step, 1)[0]
        mutation_key            = random.split(master_key + step, number_of_particles)

        #Do a SMC step
        samples, log_weights, ess   = step_for(samples, beta_next, beta_prev,weights, resampling_key, mutation_key)
        # logZ, dlogZ                 = compute_logZ(log_weights,  ess, previous_logZ, previous_dlogZ,)
        # previous_logZ               = logZ
        # previous_dlogZ              = dlogZ
        # print("ess = {}".format(ess))
        if jnp.isnan(ess):
            print("ESS is NaN, stopping SMC.")
            break

        #Store SMC step results
        smc_dict[step]["samples"]           = np.array(samples).tolist()
        smc_dict[step]["log_weights"]       = np.array(log_weights).tolist()
        smc_dict[step]["ess"]               = float(ess)
        smc_dict[step]['log_likelihoods']   = np.array(vmapped_likelihood(samples)).tolist()
        smc_dict[step]['beta']              = float(beta_next)
        beta_prev                           = beta_next
        step                               += 1
        print("Completed step {}, beta = {:.4f}, ESS = {:.2f}, ".format(step, beta_next, ess, ), end = "\r", flush = True)


    #compute evidence and draw iid samples using rejection sampling
    posterior_samples       = draw_iid_samples(smc_dict)
    print("the number of samples after rejection sampling is:", len(posterior_samples))
    logZ, dlogZ             = compute_evidence(smc_dict)
    print("logZ = {}, dlogZ = {}".format(logZ, dlogZ))


    #save results
    result_dict = {}
    result_dict['SMC']      = smc_dict
    result_dict['logZ']     = float(logZ)
    result_dict["dlogZ"]    = float(dlogZ)
    result_dict['posterior_samples'] = posterior_samples.tolist()

    # with open(f"{folder}/{label}_result.json", "w") as f:
    #     json.dump(result_dict, f)
    
    return result_dict





















