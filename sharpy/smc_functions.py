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
from jax.scipy.special import logsumexp
import os
from sharpy.utils import local_knn_covariances
from sharpy.transform import (
    from_samples_to_probit,
    from_probit_to_samples,
    from_samples_to_logit,
    from_logit_to_samples,
    log_abs_det_jacobian_probit_to_samples_per_dim,
    log_abs_det_jacobian_logit_to_samples_per_dim,
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SHARPy")




def build_mass_matrix_fn_hessian(log_posterior):
    #build mass matrix function
    def single(pos, beta):
        logdensity = lambda x: log_posterior(x, beta)
        return compute_mass_matrix(logdensity, pos)
    #use vmap_chunked to avoid OOM for large number of particles
    return jax.jit(vmap_chunked(single, in_axes=(0, None),chunk_size = 100, axis_0_is_sharded=False)) 


def build_mass_matrix_fn_knn(log_posterior, k=100):
    #build mass matrix function using knn
    def single(pos, beta):
        _, _, mass_matrix= local_knn_covariances(pos, k=k)
        # use the same mass matrix for all particles, as in knn-smc
        mass_matrices = jnp.broadcast_to(mass_matrix, (pos.shape[0], pos.shape[1], pos.shape[1]))

        return mass_matrices
    return single



def mutation_step_fn(init_fn, kernel_fn,log_posterior):
    #build mutation step with NUTS kernel
    def mutation_step(position, keys, beta, matrices):

        logdensity_fn   = lambda x: log_posterior(x, beta)  # Only for init, not passed into JIT
        # Initialize state
        state           = init_fn(position, logdensity_fn,)
        beta_batch      = jnp.broadcast_to(beta, (position.shape[0],))
        state, info       = kernel_fn(keys, state,  beta_batch, matrices)

        return state.position, info
    
    return mutation_step


def build_kernel_fn(kernel, log_posterior, step_size,gradient_based_kernel, num_integration_steps=10,max_num_doublings=6):
    

    def _kernel(rng_key, state, beta, metric):
        logdensity_fn = lambda x: log_posterior(x, beta)

        if gradient_based_kernel == "nuts":

            state, info = kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                metric,
                max_num_doublings=max_num_doublings, 
            )

            return state, info
        

        if gradient_based_kernel == "hmc":

            state, info = kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                metric,
                num_integration_steps = num_integration_steps
            )

            return state, info
    



    batched_kernel = jax.jit(
        vmap_chunked(
            _kernel,
            in_axes=(0, 0, 0, 0),
            chunk_size=1000,
            axis_0_is_sharded=False,
        )
    )

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


@jax.jit
def systematic_resample(key, weights):
    n = weights.shape[0]

    weights = weights / jnp.sum(weights)
    cdf = jnp.cumsum(weights)

    u0 = jax.random.uniform(key, minval=0.0, maxval=1.0 / n)
    positions = u0 + jnp.arange(n) / n

    idx = jnp.searchsorted(cdf, positions, side="right")
    idx = jnp.minimum(idx, n - 1)

    return idx


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


def smc_step_fn(compute_weight_and_ess):

    def smc_step(samples, beta, beta_prev, resampling_key):

        log_weights, ess = compute_weight_and_ess(samples, beta, beta_prev)

        weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))

        index = systematic_resample(resampling_key, weights)

        samples = samples[index]

        return samples, log_weights, ess, index

    return jax.jit(smc_step)










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
            n_nuts_steps        = 1,
            bounds_transform    = "unit",
            probit_bounds       = (-8.0, 8.0),
            logit_bounds        = (-8.0, 8.0),
            mass_matrix_methods = "hessian", # or knn
            gradient_based_kernel = "nuts", 

            ):
    


    if not os.path.exists(folder):
        os.makedirs(folder)


    prior_bounds = jnp.asarray(prior_bounds)
    boundary_conditions = jnp.asarray(boundary_conditions)

    #we sample in the unit cube and transform to the original space using the prior bounds
    def prior_transform(u):
        u = jnp.asarray(u)
        lo = prior_bounds[:, 0]
        hi = prior_bounds[:, 1]
        return lo + u * (hi - lo)
    
    def inverse_prior_transform(theta):
        theta = jnp.asarray(theta)
        lo = prior_bounds[:, 0]
        hi = prior_bounds[:, 1]
        return (theta - lo) / (hi - lo)

    bounds_transform = bounds_transform.lower()
    if bounds_transform not in ("unit", "probit", "logit"):
        raise ValueError("bounds_transform must be 'unit', 'probit', or 'logit'.")

    n_dim = prior_bounds.shape[0]
    if boundary_conditions.shape[0] != n_dim:
        raise ValueError("boundary_conditions must have one entry per dimension.")

    periodic_mask = boundary_conditions.astype(bool)

    # u-bounds (cube)
    prior_bounds_unit = jnp.broadcast_to(jnp.array([0.0, 1.0]), (n_dim, 2))
    prior_bounds_probit = jnp.broadcast_to(jnp.asarray(probit_bounds), (n_dim, 2))
    prior_bounds_logit = jnp.broadcast_to(jnp.asarray(logit_bounds), (n_dim, 2))


    if bounds_transform == "unit":
        sampling_bounds = prior_bounds_unit
    elif bounds_transform == "probit":
        sampling_bounds = jnp.where(
            periodic_mask[:, None],
            prior_bounds_unit,
            prior_bounds_probit,
        )
    else:
        sampling_bounds = jnp.where(
            periodic_mask[:, None],
            prior_bounds_unit,
            prior_bounds_logit,
        )


    def to_sampling_space(theta):
        """
        Physical-space theta -> sampler coordinates.

        Periodic dimensions stay in unit coordinates for transformed runs.
        """
        u = inverse_prior_transform(theta)

        if bounds_transform == "unit":
            return u

        if bounds_transform == "probit":
            z = from_samples_to_probit(theta, prior_bounds, eps=1e-12)
        else:
            z = from_samples_to_logit(theta, prior_bounds, eps=1e-12)

        return jnp.where(periodic_mask, u, z)


    def to_physical_space(position):
        """
        Sampler coordinates -> physical-space theta.

        Periodic dimensions are interpreted as unit coordinates for
        transformed runs.
        """
        if bounds_transform == "unit":
            return prior_transform(position)

        theta_unit = prior_transform(position)

        if bounds_transform == "probit":
            theta_transformed = from_probit_to_samples(position, prior_bounds)
        else:
            theta_transformed = from_logit_to_samples(position, prior_bounds)

        return jnp.where(periodic_mask, theta_unit, theta_transformed)


    def log_abs_det_jacobian_sampling_to_physical(position):
        position = jnp.asarray(position)
        lo = prior_bounds[:, 0]
        hi = prior_bounds[:, 1]
        log_width = jnp.log(hi - lo)

        if bounds_transform == "unit":
            log_det_per_dim = jnp.broadcast_to(log_width, position.shape)
        elif bounds_transform == "probit":
            log_det_per_dim = log_abs_det_jacobian_probit_to_samples_per_dim(
                position,
                prior_bounds,
            )
        else:
            log_det_per_dim = log_abs_det_jacobian_logit_to_samples_per_dim(
                position,
                prior_bounds,
            )

        log_det_per_dim = jnp.where(
            periodic_mask,
            log_width,
            log_det_per_dim,
        )
        return jnp.sum(log_det_per_dim, axis=-1)


    def log_likelihood_sampling(position):
        theta = to_physical_space(position)
        return log_likelihood(theta)


    def log_prior_sampling(position):
        theta = to_physical_space(position)
        return prior(theta) + log_abs_det_jacobian_sampling_to_physical(position)


    def log_posterior_sampling(position, beta=1.0):
        return log_prior_sampling(position) + beta * log_likelihood_sampling(position)

    #Set up the SMC components
    if gradient_based_kernel == "nuts":

        init_fn = jax.vmap(blackjax.nuts.init, in_axes=(0, None))
        kernel = blackjax.nuts.build_kernel(
            sampling_bounds,
            boundary_conditions,
            integrators.velocity_verlet,
            divergence_threshold=100,
        )
    if gradient_based_kernel == "hmc":
        init_fn = jax.vmap(blackjax.hmc.init, in_axes=(0, None))
        kernel = blackjax.hmc.build_kernel(
            sampling_bounds,
            boundary_conditions,
            integrators.velocity_verlet,
            divergence_threshold=100,
        )



    if mass_matrix_methods == "hessian":

         mass_matrix_fn = build_mass_matrix_fn_hessian(log_posterior_sampling)
    elif mass_matrix_methods == "knn":
        mass_matrix_fn = build_mass_matrix_fn_knn(log_posterior_sampling, k=1000)


    compute_weight_and_ess = compute_weight_and_ess_fn(log_likelihood_sampling)



    

    vmapped_likelihood_sampling = jax.jit(jax.vmap(log_likelihood_sampling))
    vmapped_prior_sampling = jax.jit(jax.vmap(log_prior_sampling))


    smc_dict                    = {}        



    kernel_fn = build_kernel_fn(
        kernel,
        log_posterior_sampling,
        step_size,
        max_num_doublings           = 6,
        num_integration_steps       = 10, 
        gradient_based_kernel       = gradient_based_kernel
    )

    mutation_step_vectorized = mutation_step_fn(
        init_fn,
        kernel_fn,
        log_posterior_sampling,
    )

    step_for = smc_step_fn(compute_weight_and_ess)
    

    if isinstance(initial_particles, str) and initial_particles == "prior":

        initial_particles, _ = sample_from_prior(master_key,number_of_particles, prior_logprob=prior, bounds=prior_bounds, oversample=5)

        # initial_position = jax.random.uniform(master_key, shape=(number_of_particles, n_dim), minval = prior_bounds[:, 0], maxval=prior_bounds[:, 1])   
        initial_position  = to_sampling_space(initial_particles)
        

    else:
        initial_position = to_sampling_space(initial_particles)

    
    
    #initialize SMC
    

    initial_beta    = 0.0
    beta_prev       = initial_beta
    # weights         = initial_weights
    samples         = initial_position
    beta_next       = initial_beta
    step            = 0

    logZ            = initial_logZ



    diagnostic = {}
    
    #SMC main loop
    while beta_next < 1.0 - 1e-8:
        beta_next = find_next_beta(compute_weight_and_ess, samples, beta_prev,ess_target= int(number_of_particles * alpha))
        # sys.exit()
        smc_dict[int(step)] = {}
        diagnostic[int(step)] = {}


        resampling_key          = random.split(master_key+42 + step, 1)[0]
        mutation_key            = random.split(master_key + step, number_of_particles)
        


        samples, log_weights, ess, index = step_for(
            samples,
            beta_next,
            beta_prev,
            resampling_key,
        )

        incremental_logZ = logsumexp(log_weights) - jnp.log(len(log_weights))
        logZ += incremental_logZ

        # Then multiple NUTS mutations
        samples_before_mutation = samples
        sub_set_samples = samples_before_mutation[::10]  # take a subset of samples to compute the mass matrix
        matrices = mass_matrix_fn(sub_set_samples, beta_next)  # compute once per SMC stage



        lambdas, V = jnp.linalg.eigh(matrices)
        median_lambda = jnp.median(lambdas, axis=0)
        # print("median_lambda", median_lambda)
        median_V  = jnp.median(V, axis=0)

        median_matrix = median_V @ jnp.diag(median_lambda) @ median_V.T
        # matrix = jnp.co
        matrices  = jnp.broadcast_to(median_matrix, (samples.shape[0], median_matrix.shape[0], median_matrix.shape[1]))

        for m in range(n_nuts_steps):
            mutation_key = random.split(
                master_key + 10_000 * step + m,
                number_of_particles,
            )

            samples, nuts_info = mutation_step_vectorized(
                samples,
                mutation_key,
                beta_next,
                matrices,
            )

        mean_acceptance_rate = float(np.mean(nuts_info.acceptance_rate))
        # mean_num_integration_steps = float(np.mean(nuts_info.num_integration_steps))
        # mean_num_trajectory_expansions = float(np.mean(nuts_info.num_trajectory_expansions))
        # num_divergent = int(jnp.count_nonzero(nuts_info.is_divergent))
        # num_turning = int(jnp.count_nonzero(nuts_info.is_turning))  
        diagnostic[int(step)]["mean_acceptance_rate"] = mean_acceptance_rate
        # diagnostic[int(step)]["mean_num_integration_steps"] = mean_num_integration_steps
        # diagnostic[int(step)]["mean_num_trajectory_expansions"] = mean_num_trajectory_expansions
        # diagnostic[int(step)]["num_divergent"] = num_divergent
        # diagnostic[int(step)]["num_turning"] = num_turning  
        diagnostic[int(step)]["logZ"] = float(logZ)
        diagnostic[int(step)]["incremental_logZ"] = float(incremental_logZ)
            # logger.info("NUTS mutation step %d for SMC step %d", m + 1, step)

            # logger.info(
            #     "mean acceptance rate = %.3f",
            #     float(np.mean(nuts_info.acceptance_rate)),
            # )

            # logger.info(
            #     "mean number integration steps = %.3f",
            #     float(np.mean(nuts_info.num_integration_steps)),
            # )

            # logger.info(
            #     "mean num_trajectory_expansions = %.3f",
            #     float(np.mean(nuts_info.num_trajectory_expansions)),
            # )

            # logger.info(
            #     "num is_divergent = %d out of %d",
            #     int(jnp.count_nonzero(nuts_info.is_divergent)),
            #     len(nuts_info.is_divergent),
            # )

            # logger.info(
            #     "num is_turning = %d out of %d",
            #     int(jnp.count_nonzero(nuts_info.is_turning)),
            #     len(nuts_info.is_turning),
            # )


        if jnp.isnan(ess):
            logger.error("ESS is NaN at step {}, beta = {:.4f}. Terminating SMC.".format(step, beta_next))
            return -1

        #Store SMC step results
        smc_dict[step]["samples"] = np.array(samples).tolist()
        smc_dict[step]["log_weights"] = np.array(log_weights).tolist()
        smc_dict[step]["ess"] = float(ess)

        smc_dict[step]["log_likelihoods"] = np.array(
            vmapped_likelihood_sampling(samples)
        ).tolist()

        smc_dict[step]["log_prior"] = np.array(
            vmapped_prior_sampling(samples)
        ).tolist()

        smc_dict[step]["beta"] = float(beta_next)
        beta_prev                           = beta_next
        step                               += 1
        # print("Completed step {}, beta = {:.4f}, ESS = {:.2f}, ".format(step, beta_next, ess, ), end = "\r", flush = True)
        logger.info("Completed step {}, beta = {:.4f}, ESS = {:.2f}, logZ = {:.2f}, mean_acceptance = {:.2f} \r".format(step, beta_next, ess, logZ, mean_acceptance_rate))

    #compute evidence and draw iid samples using rejection sampling
    posterior_samples       = draw_iid_samples(smc_dict)
    logger.info("i.i.d samples and evidence using only SMC samples")
    logger.info("The number of samples after rejection sampling is: {}".format(len(posterior_samples)))
    logZ, dlogZ             = compute_evidence(smc_dict, initial_logZ = initial_logZ)
    logger.info("Estimated log-evidence: {:.4f} ± {:.4f}".format(logZ, dlogZ))

    #save diagnostics plots
    diagnostic_folder = f"{folder}/diagnostic_plots"
    if not os.path.exists(diagnostic_folder):
        os.makedirs(diagnostic_folder)
    # Plot diagnostics
    import matplotlib.pyplot as plt
    number_of_steps = len(smc_dict)
    steps = np.arange(number_of_steps)
    betas = [smc_dict[key]["beta"] for key in smc_dict.keys()]
    ess_values = [smc_dict[key]["ess"] for key in smc_dict.keys()]
    # num_divergent = [diagnostic[key]["num_divergent"] for key in diagnostic.keys()]
    # num_turning = [diagnostic[key]["num_turning"] for key in diagnostic.keys()]
    mean_acceptance_rate = [diagnostic[key]["mean_acceptance_rate"] for key in diagnostic.keys()]
    # mean_num_integration_steps = [diagnostic[key]["mean_num_integration_steps"] for key in diagnostic.keys()]
    # mean_num_trajectory_expansions = [diagnostic[key]["mean_num_trajectory_expansions"] for key in diagnostic.keys()]   
    fig, axes = plt.subplots(9, 1, figsize=(10, 30))
    axes[0].plot(steps, betas, marker="o")
    axes[0].set_title("Beta schedule")
    axes[0].set_xlabel("SMC step")
    axes[0].set_ylabel("Beta")
    axes[1].plot(steps, ess_values, marker="o")
    axes[1].set_title("Effective Sample Size (ESS)")
    axes[1].set_xlabel("SMC step")
    axes[1].set_ylabel("ESS")
    # axes[2].plot(steps, num_divergent, marker="o")
    # axes[2].set_title("Number of Divergent Transitions")    
    # axes[2].set_xlabel("SMC step")
    # axes[2].set_ylabel("Number of Divergent Transitions")
    # axes[3].plot(steps, num_turning, marker="o")
    # axes[3].set_title("Number of Turning Transitions")    
    # axes[3].set_xlabel("SMC step")
    # axes[3].set_ylabel("Number of Turning Transitions")
    axes[4].plot(steps, mean_acceptance_rate, marker="o")
    axes[4].set_title("Mean Acceptance Rate")    
    axes[4].set_xlabel("SMC step")
    axes[4].set_ylabel("Mean Acceptance Rate")
    # axes[5].plot(steps, mean_num_integration_steps, marker="o")
    # axes[5].set_title("Mean Number of Integration Steps")    
    # axes[5].set_xlabel("SMC step")
    # axes[5].set_ylabel("Mean Number of Integration Steps")
    # axes[6].plot(steps, mean_num_trajectory_expansions, marker="o")
    # axes[6].set_title("Mean Number of Trajectory Expansions")    
    # axes[6].set_xlabel("SMC step")
    # axes[6].set_ylabel("Mean Number of Trajectory Expansions")
    axes[7].plot(steps, [diagnostic[key]["logZ"] for key in diagnostic.keys()], marker="o")
    axes[7].set_title("Estimated log-evidence at each SMC step")
    axes[7].set_xlabel("SMC step")
    axes[7].set_ylabel("Estimated log-evidence")
    axes[8].plot(steps, [diagnostic[key]["incremental_logZ"] for key in diagnostic.keys()], marker="o")
    axes[8].set_title("Incremental log-evidence at each SMC step")
    axes[8].set_xlabel("SMC step")
    axes[8].set_ylabel("Incremental log-evidence")  
    plt.tight_layout()
    plt.savefig(f"{diagnostic_folder}/{label}_diagnostics.png")
    plt.close()


    if use_flow:
        logger.info("Fitting normalizing flow to final SMC samples...")
        from coppuccino import normalizing_flows_fit, sample, log_prob
        

        posterior_samples_from_iterations = np.array(
            [smc_dict[key]["samples"] for key in list(smc_dict.keys())[-3:]]
        )
        posterior_samples_from_iterations = np.concatenate(
            posterior_samples_from_iterations, axis=0
        )

        samples_theta = to_physical_space(posterior_samples_from_iterations)

        flow = normalizing_flows_fit(
            samples_theta,
            max_epochs=200,
            rng_seed=42,
            prior_bounds=prior_bounds,
        )

        # Fresh physical-space flow samples
        new_samples = sample(flow, n_samples=1000, rng_seed=42)

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
    result_dict["posterior_samples"] = np.array(
        to_physical_space(jnp.asarray(posterior_samples))
    ).tolist()
    result_dict["bounds_transform"] = bounds_transform
    result_dict["sampling_space_posterior_samples"] = np.array(posterior_samples).tolist()
    result_dict[f"{bounds_transform}_posterior_samples"] = np.array(posterior_samples).tolist()
    result_dict['resampled_samples']        = resampled_samples.tolist() if use_flow else None

    with open(f"{folder}/{label}_result.json", "w") as f:
        json.dump(result_dict, f)
    
    return result_dict























