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
def multinomial_resample(key, particles, weights, ):
    cdf = jnp.cumsum(weights)
    u = jax.random.uniform(key, shape=(len(weights),))
    idx = jnp.searchsorted(cdf, u)
    return particles[idx]


@jax.jit
def multinomial_resample_PS(key, particles, weights,):
    cdf = jnp.cumsum(weights)
    u = jax.random.uniform(key, shape=(1000),)
    idx = jnp.searchsorted(cdf, u)
    return particles[idx]



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

        # Resampling
        samples         = multinomial_resample(resampling_key, samples, weights)
        # Mutation
        matrices        = mass_matrix_fn(samples, beta)
        
        samples         = mutation_step_vectorized(samples, mutation_keys, beta, matrices)

        return samples, weights, weights_nonorm, ess
    
    
    
    return smc_step











def run_smc(log_likelihood, prior, prior_bounds, boundary_conditions, temperature_schedule, number_of_particles, step_size,   master_key):

    def log_posterior(params, beta=1):
        return log_likelihood(params)*beta + prior(params)

    kernel                      = blackjax.nuts.build_kernel( prior_bounds, boundary_conditions, integrators.velocity_verlet)
    
    mass_matrix_fn              = build_mass_matrix_fn(log_posterior)
    kernel_fn                   = build_kernel_fn(kernel, log_posterior, step_size)
    compute_weight_and_ess      = compute_weight_and_ess_fn(log_likelihood)
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



# class Particle:
#     def __init__(self, position,log_likelihood, beta):
#         self.position       = position
#         self.beta           = beta
#         self.log_likelihood = log_likelihood

# class Step:
#     def __init__(self, beta, particles, evidence, evidence_error):

#         self.beta                = beta
#         self.particles           = particles
#         self.evidence            = evidence
#         self.evidence_error      = evidence_error
#         self.number_of_particles = len(particles)


    
#     def get_positions(self):
#         return np.array([p.position for p in self.particles])
    
#     def get_log_likelihoods(self):
#         return np.array([p.log_likelihood for p in self.particles])
    


        
from scipy.special import logsumexp



def compute_persistent_weights(particles,number_of_steps,current_beta  ):
        
        
       
        ## particle: (position, log_likelihood, beta, log_Z)
        
        
        log_likelihoods     = particles[:,2]
        betas               = particles[:,3]
        log_numerator       = log_likelihoods *current_beta
        
        
        evidences           = particles[:,4]
        beta                = np.unique(betas)
        print("beta, ", beta)
        evidence            = np.unique(evidences)
        print("loglike_shape", log_likelihoods.shape, )

        log_numerator           = log_numerator
        log_denominator         = np.array([log_likelihoods *beta[i] - evidence[i] for i in range(len(beta))])
        print("log_denominator_shape" , log_denominator.shape)
        
        log_denominator         = np.logaddexp.reduce(log_denominator, ) - np.log(len(beta))
        print("log_denominator_shape" , log_denominator.shape)
        log_weights             = log_numerator - log_denominator
        # log_weights             = np.concatenate(log_weights)
        log_z                   = np.logaddexp.reduce(log_weights) - np.log(len(log_weights))   
        #log_weights             = log_weights - np.logaddexp.reduce(log_weights)


        # log_piece_to_sum    = log_likelihoods*betas  - evidences

        # print(log_piece_to_sum)

        # print("number_of_steps = {}".format(number_of_steps))   
        # log_denominator     = -np.log(number_of_steps-1) + logsumexp(log_piece_to_sum, axis=0)
        # # print(log_denominator)
        # log_weights         = log_numerator - log_denominator
        # log_z               = np.logaddexp.reduce(log_weights) - np.log(len(log_weights))
        # log_weights         = log_weights - np.logaddexp.reduce(log_weights)
        
        # betas              = np.unique(particles[:,3])
        
        # ## weights corresponding to each beta
        # filtered_weights   = [log_weights[particles[:,3] == b] for b in betas]
        # for i, l in enumerate(filtered_weights):
        #     print(len(l))
        
        # # print("filtered_weights = {}".format(filtered_weights)) 

        # #compute eq. 16 in https://arxiv.org/pdf/2407.20722
        # log_sum_evidence_piece = [logsumexp(weights_beta) - np.log(len(weights_beta)) for weights_beta in filtered_weights]
        # log_z                  = logsumexp(np.exp(log_sum_evidence_piece)) -  np.log(number_of_steps-1)

        # log_Z                 = logsumexp(log_piece_to_sum) 

        return log_weights, log_z




        



        


        





def run_persistent_smc(log_likelihood, 
                        prior,
                        prior_bounds,
                        boundary_conditions,
                        temperature_schedule, 
                        number_of_particles, 
                        step_size,   
                        master_key):
    
        
    def log_posterior(params, beta=1):
        return log_likelihood(params)*beta + prior(params)
    
    kernel                      = blackjax.nuts.build_kernel( prior_bounds, boundary_conditions, integrators.velocity_verlet)
    mass_matrix_fn              = build_mass_matrix_fn(log_posterior)
    kernel_fn                   = build_kernel_fn(kernel, log_posterior, step_size)
    compute_weight_and_ess      = compute_weight_and_ess_fn(log_likelihood)
    init_fn                     = (jax.vmap(blackjax.nuts.init, in_axes=(0, None, )))
    mutation_step_vectorized    = mutation_step_fn(init_fn, kernel_fn, log_posterior)

    initial_position            = jax.random.uniform(
                                                    jax.random.PRNGKey(1),
                                                    shape=(number_of_particles, len(prior_bounds)),
                                                    minval=prior_bounds[:, 0],
                                                    maxval=prior_bounds[:, 1]
                                                    )
    
    log_likelihoods             = jax.vmap(log_likelihood,)(initial_position)

    n_steps                     = len(temperature_schedule)

    mutation_keys               = random.split(master_key,(n_steps, number_of_particles))
    resampling_keys             = random.split(master_key+42, n_steps)
    beta_prev                   = 0.0
    number_of_step              = 1

    # print(len(initial_position), len(log_likelihoods), len(np.ones(number_of_particles)*beta_prev), len(np.zeros(number_of_particles)))
    # print(log_likelihoods.shape)
    particles                   = np.column_stack((initial_position, log_likelihoods, np.ones(number_of_particles)*beta_prev, np.zeros(number_of_particles)))

    
    
    ## particle: (position, log_likelihood, beta, log_Z)
    for step, beta in enumerate(temperature_schedule):
        beta = np.array(beta)
        print("Step: {}, beta = {}".format(step, beta))
        resampling_key          = resampling_keys[step]
        mutation_key            = mutation_keys[step]
        number_of_step         += 1
        log_weights, logZ                = compute_persistent_weights(particles, number_of_step, beta)
        
        # normalized_log_weights          = log_weights - logZ
        weights                         = np.exp(log_weights)
        weights                         = weights / np.sum(weights)
        # ess                             = (np.sum(weights)) ** 2 / np.sum(weights**2)
        ess                            = 1/np.sum(weights**2) 
        # print("weights sum = {}".format((weights)))
        particle_position               = particles[:,:2]
     
        
        resampled_particles             = jax.random.choice(resampling_key, particle_position, (number_of_particles,), p=weights)

        # print("resampled_particles = {}".format(resampled_particles))
        # print(particles[])

        matrices                        = mass_matrix_fn(resampled_particles, beta)
        mutated_samples                 = mutation_step_vectorized(resampled_particles, mutation_key, beta, matrices)
        
        log_likelihoods                 = jax.vmap(log_likelihood,)(mutated_samples)
        # print("mutated_samples = {}".format(log_likelihoods.shape))
        # print(np.ones(number_of_particles)*beta)
        new_particles                    = np.column_stack((mutated_samples, log_likelihoods, np.ones(number_of_particles)*beta, np.ones(number_of_particles)*logZ))

        particles                        = np.row_stack((particles, new_particles))
        print("ess = {}".format(ess))
        print("logZ = {}".format(logZ))


    return particles, weights


        












        

    






    






    return 



def compute_evidence(folder, label):
    import json
    result_path = f'{folder}/{label}/result.json'
    with open(result_path, 'r') as f:
        result = json.load(f)
    evidence = 1.0
    errors    = []
    

    for key in result.keys():
        if float(key) > 0:
            
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







def find_global_minimum_nuts(log_posterior, prior_bounds, boundary_conditions,  number_of_samples_single_chain, number_of_parallel_chains,  step_size,  ):


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



    





def find_global_minimum_jaxopt(log_posterior, prior_bounds ):   
    import jax
    import jax.numpy as jnp
    import jaxopt

    a  = prior_bounds[:,0]
    b  = prior_bounds[:,1]
    
    def y_to_x(y):
        s = jax.nn.sigmoid(y)       # (0,1)
        return a + (b - a) * s      # (a,b)

    def fun_y(y):
        return log_posterior(y_to_x(y))
    # Local Newton solver (uses gradients/Hessians automatically)
    solver = jaxopt.LBFGS(fun=fun_y, maxiter=200, tol=1e-6)

    # One local run
    def run_one(x0):
        result = solver.run(init_params=x0)
        return result.params, result.state.value
    


    # Vectorized over multiple starts
    batched_run = jax.vmap(run_one, in_axes=0, out_axes=(0, 0))

    def global_optimize(rng, n_starts=64):
        rng, subkey = jax.random.split(rng)
        x0s = jax.random.uniform(subkey, shape=(n_starts, len(prior_bounds)),minval=prior_bounds[:, 0],
                                                              maxval=prior_bounds[:, 1])
        
        xs, vals = batched_run(x0s)  # run all in parallel
        best_idx = jnp.argmin(vals)
        return xs[best_idx], vals[best_idx]

    # Run
    rng = jax.random.PRNGKey(0)
    xopt, fopt = global_optimize(rng, n_starts=10)

    def x_to_y(x):
    # avoid division by zero with clip
        s = (x - a) / (b - a)
        s = jnp.clip(s, 1e-12, 1 - 1e-12)
        return jnp.log(s) - jnp.log1p(-s) 

    y_opt = x_to_y(xopt)

    print("Best solution:", y_opt)
    print("Best value:", fopt)
    return xopt, fopt