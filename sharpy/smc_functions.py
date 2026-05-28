import jax
from sharpy.utils import compute_mass_matrix, sample_from_prior
import jax.numpy as jnp 
from jax import random
from blackjax.mcmc import integrators
import blackjax
import numpy as np 
# from netket.jax import vmap_chunked
from sharpy.utils import vmap_chunked
import json
from corner import corner
import os

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SHARPy")




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
        return kernel(rng_key, state, logdensity_fn, step_size, metric, max_num_doublings=8)
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
    def smc_step(samples, beta,beta_prev, resampling_key, mutation_keys):

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
    log_priors      = []
    betas           = []
    log_evidences   = []
    log_evidence    = 0.0 #this is the evidence of the prior 

    for key in result.keys():
        
        samples         += list(result[key]['samples'])
        # log_posteriors += list((result[key]['log_posteriors']))
        log_likelihoods += list(result[key]['log_likelihoods'])
        log_priors      += list(result[key]['log_prior'])
        
        betas.append(result[key]['beta'])

        log_evidence_piece = logsumexp(result[key]['log_weights']) - np.log(len(result[key]['log_weights']))
            
        log_evidence      += log_evidence_piece
        log_evidences.append(log_evidence)
        
    betas                   = np.array(betas)
    log_evidences           = np.array(log_evidences)
    samples                 = np.array(samples)
    log_likelihoods         = np.array(log_likelihoods)
    log_priors              = np.array(log_priors)
    log_posteriors          = np.array(log_likelihoods) + np.array(log_priors)


    log_posterior_primed    = np.array([
                                        beta * log_likelihoods + log_priors - log_evidence
                                        for beta, log_evidence in zip(betas, log_evidences)
                                    ])
   
    log_posterior_primed        = jnp.logaddexp.reduce( log_posterior_primed, axis = 0) - jnp.log(len(result.keys()))

  
    #rejection sampling
    M           = np.max( log_posteriors - log_posterior_primed)  
    u           = np.random.uniform( size = len(log_posterior_primed))
    accepted    =  +log_posteriors - log_posterior_primed - M > np.log(u)
    samples     = samples[accepted]

    

    return samples



def compute_evidence(result_dict, initial_logZ = 0.0):

    
    log_evidence = initial_logZ
    errors       = []
    
    
    

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
        ess_diff = compute_weight_and_ess(samples, beta_next, beta_prev)[1] - ess_target
        if ess_diff > 0:
            beta_lower = beta_next
        else:
            beta_upper= beta_next
    return beta_next







def run_sharpy(log_likelihood, 
            prior, 
            prior_bounds,
            boundary_conditions, 
            alpha,
            number_of_particles, 
            step_size,   
            master_key,
            folder              = ".",
            label               = "run",
            initial_particles   = "prior",
            initial_logZ        = 0.0,
            initial_dlogZ       = 0.0,
            use_flow            = False,
            ):
    


    if not os.path.exists(folder):
        os.makedirs(folder)


    #we sample in the unit cube and transform to the original space using the prior bounds
    def prior_transform(u):
        lo = prior_bounds[:, 0]
        hi = prior_bounds[:, 1]
        return lo + u * (hi - lo)
    
    def inverse_prior_transform(theta):
        lo = prior_bounds[:, 0]
        hi = prior_bounds[:, 1]
        return (theta - lo) / (hi - lo)

    # u-bounds (cube)
    prior_bounds_unit = jnp.array([[0.0, 1.0]] * prior_bounds.shape[0])

    def log_posterior_unit(u, beta=1.0):
        q = prior_transform(u)
        return beta * log_likelihood(q) + prior(q)
    
    def log_likelihood_unit(u):
        theta = prior_transform(u)
        return log_likelihood(theta)
    

    def log_prior_unit(u):
        theta = prior_transform(u)
        return prior(theta)
    

    def log_abs_det_jacobian_prior_transform(u):
        lo = prior_bounds[:, 0]
        hi = prior_bounds[:, 1]
        return jnp.sum(jnp.log(hi - lo))


    def log_likelihood_unit(u):
        theta = prior_transform(u)
        return log_likelihood(theta)


    def log_prior_unit(u):
        theta = prior_transform(u)
        return prior(theta) + log_abs_det_jacobian_prior_transform(u)


    def log_posterior_unit(u, beta=1.0):
        return log_prior_unit(u) + beta * log_likelihood_unit(u)

    #Set up the SMC components
    kernel                      = blackjax.nuts.build_kernel( prior_bounds_unit, boundary_conditions, integrators.velocity_verlet, divergence_threshold=100 )
    mass_matrix_fn              = build_mass_matrix_fn(log_posterior_unit, )
    kernel_fn                   = build_kernel_fn(kernel, log_posterior_unit, step_size)
    compute_weight_and_ess      = compute_weight_and_ess_fn(log_likelihood_unit)
    init_fn                     = (jax.vmap(blackjax.nuts.init, in_axes=(0, None, )))
    mutation_step_vectorized    = mutation_step_fn(init_fn, kernel_fn, log_posterior_unit)
    step_for                    = smc_step_fn(mass_matrix_fn, mutation_step_vectorized, compute_weight_and_ess, )
    vmapped_likelihood_unit     = jax.jit(jax.vmap(log_likelihood_unit))
    vmapped_prior_unit          = jax.jit(jax.vmap(log_prior_unit))
    vmapped_posterior_unit       = jax.jit(jax.vmap(log_posterior_unit))
    smc_dict                    = {}        

    

    #Generate initial particles from the prior
    if initial_particles == "prior":

        initial_position, initial_logZ = sample_from_prior(jax.random.PRNGKey(1), number_of_particles, log_prior_unit, prior_bounds_unit, oversample=5)

        initial_logZ = 0
    else:
        initial_position = initial_particles
        

    
    
    #initialize SMC
    

    initial_beta    = 0.0
    beta_prev       = initial_beta
    # weights         = initial_weights
    samples         = initial_position
    beta_next       = initial_beta
    step            = 0




    
    #SMC main loop
    while beta_next < 1.0 - 1e-8:
        beta_next = find_next_beta(compute_weight_and_ess, samples, beta_prev,ess_target= int(number_of_particles * alpha))
        # sys.exit()
        smc_dict[int(step)] = {}


        resampling_key          = random.split(master_key+42 + step, 1)[0]
        mutation_key            = random.split(master_key + step, number_of_particles)

        #Do a SMC step
        samples, log_weights, ess   = step_for(samples, beta_next, beta_prev, resampling_key, mutation_key)
      
        if jnp.isnan(ess):
            logger.error("ESS is NaN at step {}, beta = {:.4f}. Terminating SMC.".format(step, beta_next))
            return -1

        #Store SMC step results
        smc_dict[step]["samples"]           = np.array(samples).tolist()
        smc_dict[step]["log_weights"]       = np.array(log_weights).tolist()
        smc_dict[step]["ess"]               = float(ess)
        smc_dict[step]['log_likelihoods']   = np.array(vmapped_likelihood_unit(samples)).tolist()
        smc_dict[step]['log_prior']         = np.array(vmapped_prior_unit(samples)).tolist()
        smc_dict[step]['beta']              = float(beta_next)
        beta_prev                           = beta_next
        step                               += 1
        # print("Completed step {}, beta = {:.4f}, ESS = {:.2f}, ".format(step, beta_next, ess, ), end = "\r", flush = True)
        logger.info("Completed step {}, beta = {:.4f}, ESS = {:.2f} \r".format(step, beta_next, ess))

    #compute evidence and draw iid samples using rejection sampling
    posterior_samples       = draw_iid_samples(smc_dict)
    logger.info("i.i.d samples and evidence using only SMC samples")
    logger.info("The number of samples after rejection sampling is: {}".format(len(posterior_samples)))
    logZ, dlogZ             = compute_evidence(smc_dict, initial_logZ = initial_logZ)
    logger.info("Estimated log-evidence: {:.4f} ± {:.4f}".format(logZ, dlogZ))

    if use_flow:
        logger.info("Fitting normalizing flow to final SMC samples...")
        from coppuccino import normalizing_flows_fit, sample, log_prob
        from jax.scipy.special import logsumexp

        posterior_samples_from_iterations = np.array(
            [smc_dict[key]["samples"] for key in list(smc_dict.keys())[-3:]]
        )
        posterior_samples_from_iterations = np.concatenate(
            posterior_samples_from_iterations, axis=0
        )

        # SMC samples are in unit cube, flow is trained in physical coordinates
        samples_theta = prior_transform(posterior_samples_from_iterations)

        flow = normalizing_flows_fit(
            samples_theta,
            max_epochs=200,
            rng_seed=42,
            prior_bounds=prior_bounds,
        )

        # Fresh physical-space flow samples
        new_samples = sample(flow, n_samples=10000, rng_seed=42)

        # Physical-space target:
        # log target(theta) = log L(theta) + log pi(theta)
        log_like_theta = jnp.ravel(jax.vmap(log_likelihood)(new_samples))
        log_prior_theta = jnp.ravel(jax.vmap(prior)(new_samples))

        log_target_theta = log_like_theta + log_prior_theta

        # Physical-space flow density.
        # Prefer coppuccino.log_prob wrapper, not flow.log_prob directly.
        logq_theta = jnp.ravel(log_prob(flow, new_samples))

        log_weights = log_target_theta - logq_theta

        N = log_weights.shape[0]

        logZ_flow = logsumexp(log_weights) - jnp.log(N)

        weights = jnp.exp(log_weights - logsumexp(log_weights))
        Ess = 1.0 / jnp.sum(weights**2)

        se_logZ = jnp.sqrt(jnp.maximum(0.0, 1.0 / Ess - 1.0 / N))

        logger.info("Effective sample size of flow samples: {:.2f} / {}".format(float(Ess), N))
        logger.info(
            "Estimated log-evidence with flow samples: {:.4f} ± {:.4f}".format(
                float(logZ_flow), float(se_logZ)
            )
        )

        index = jax.random.choice(
            random.PRNGKey(123),
            jnp.arange(N),
            shape=(int(jnp.floor(Ess)),),
            replace=True,
            p=weights,
        )

        resampled_samples = new_samples[index]





    #save results
    result_dict = {}
    result_dict['SMC']      = smc_dict
    result_dict['logZ']     = float(logZ)
    result_dict["dlogZ"]    = float(dlogZ)
    result_dict['posterior_samples']        = prior_transform(posterior_samples).tolist()
    result_dict['resampled_samples']        = resampled_samples.tolist() if use_flow else None

    with open(f"{folder}/{label}_result.json", "w") as f:
        json.dump(result_dict, f)
    
    return result_dict


























