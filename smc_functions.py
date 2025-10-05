import jax
from utils import compute_mass_matrix
import jax.numpy as jnp 
from jax import random
from blackjax.mcmc import integrators
import blackjax

import numpy as np 


from netket.jax import vmap_chunked


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


# # @jax.jit
# def multinomial_resample_PS(key, particles, weights,):
#     cdf = jnp.cumsum(weights)
#     u = jax.random.uniform(key, shape=(1000),)
#     idx = jnp.searchsorted(cdf, u)
#     return particles[idx]



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

        # Resampling
        # samples         = multinomial_resample(resampling_key, samples, weights)
        samples         = jax.random.choice(resampling_key, samples, (len(samples),), p=weights)
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
    vmapped_likelihood          = jax.jit(jax.vmap(log_likelihood))
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
        if jnp.isnan(ess):
            print("ESS is NaN, stopping SMC.")
            break
        samples_dict[step]["samples"]           = np.array(samples).tolist()
        samples_dict[step]["weights"]           = np.array(weights).tolist()
        samples_dict[step]["ess"]               = float(ess)
        samples_dict[step]['log_likelihoods']   = np.array(vmapped_likelihood(samples)).tolist()
        samples_dict[step]['beta']              = float(beta)


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








        



        


        





def run_persistent_smc(log_likelihood, 
                        prior,
                        prior_bounds,
                        boundary_conditions,
                        temperature_schedule, 
                        number_of_particles, 
                        step_size,   
                        master_key):
    

    dimension = len(prior_bounds)
    def log_posterior(params, beta=1):
        return log_likelihood(params)*beta + prior(params)
    
    kernel                      = blackjax.nuts.build_kernel( prior_bounds, boundary_conditions, integrators.velocity_verlet)
    mass_matrix_fn              = build_mass_matrix_fn(log_posterior)
    kernel_fn                   = build_kernel_fn(kernel, log_posterior, step_size)
    init_fn                     = (jax.vmap(blackjax.nuts.init, in_axes=(0, None, )))
    mutation_step_vectorized    = mutation_step_fn(init_fn, kernel_fn, log_posterior)
    vmapped_likelihood          = jax.jit(jax.vmap(log_likelihood))

    initial_position            = jax.random.uniform(
                                                    master_key,
                                                    shape=(number_of_particles, len(prior_bounds)),
                                                    minval=prior_bounds[:, 0],
                                                    maxval=prior_bounds[:, 1]
                                                    )
    

    log_likelihoods             = vmapped_likelihood(initial_position)

    n_steps                     = len(temperature_schedule)

    mutation_keys               = random.split(master_key,(n_steps, number_of_particles))
    resampling_keys             = random.split(master_key+42, n_steps)
    beta_prev                   = 0.0
    


    particles                   = jnp.column_stack((initial_position,
                                                    log_likelihoods,
                                                    np.ones(number_of_particles)*beta_prev, 
                                                    np.zeros(number_of_particles)   )
                                                    )


    log_z_variances            = []
    for step, beta in enumerate(temperature_schedule):
        
        beta                            = jnp.array(beta)
        print("Step: {}, ".format(step,))
        print("Beta: {}, ".format(beta,))

        resampling_key                  = resampling_keys[step]
        mutation_key                    = mutation_keys[step]
      
        log_weights, logZ, logZerr      = compute_persistent_weights(particles, beta, dimension)
       

        
        weights                         = jnp.exp(log_weights)
        weights                         = weights / jnp.sum(weights)
        ess                             = 1       / jnp.sum(weights**2) 

        particle_position               = particles[:,:dimension -1+1]

        
        
     
        #remsapling
        resample_indexes                = (jax.random.choice)(resampling_key, jnp.arange(len(particle_position)), (len(initial_position),), p=weights)
        resampled_particles             = particle_position[resample_indexes]
        # resampled_particles             = multinomial_resample(resampling_key, particle_position, weights)
       
        #mutation
        matrices                        = mass_matrix_fn(resampled_particles, beta)
        mutated_samples                 = mutation_step_vectorized(resampled_particles, mutation_key, beta, matrices)
        log_likelihoods                 = vmapped_likelihood(mutated_samples)



        new_particles                   = jnp.column_stack((mutated_samples, log_likelihoods,jnp.ones(number_of_particles)*beta, jnp.ones(number_of_particles)*logZ))
        particles                       = jnp.vstack((particles, new_particles))
        

        print("ess = {}".format(ess))
        print("logZ = {}".format(logZ))
        log_z_variances.append(logZerr)
        
        logZerr = jnp.sqrt(np.sum(np.cumsum(log_z_variances)))
        print("logZerr = {}".format(logZerr))


        


    return particles, weights, logZ, logZerr


def compute_unique(arr):

    res,ind = np.unique(arr, return_index=True)
    result = res[np.argsort(ind)]
    return result


def compute_persistent_weights(particles, current_beta, dimension,):
        


        evidences               = particles[:,dimension -1 +3]
        betas                   = particles[:,dimension -1 +2]    
        beta                    = compute_unique(betas)
        evidence                = compute_unique(evidences)
        log_likelihoods         = particles[:,dimension -1 +1]
        
        log_weights, log_z      = compute_log_weights_and_log_z(log_likelihoods, beta, evidence, current_beta)
           
        
        log_z_var               = compute_bootstrap_variance(jax.random.PRNGKey(0), log_weights)

        log_weights             = log_weights - jnp.logaddexp.reduce(log_weights)


        return log_weights, log_z, log_z_var

def draw_iid_posterior_samples(particles, dimension):

    evidences               = particles[:,dimension -1 +3]
    betas                   = particles[:,dimension -1 +2]    
    beta                    = compute_unique(betas)
    evidence                = compute_unique(evidences)
    log_likelihoods         = particles[:,dimension -1 +1]
    log_posterior_primed        = log_likelihoods * beta[:, None] - evidence[:, None]
    log_posterior_primed        = jnp.logaddexp.reduce( log_posterior_primed, axis = 0) #- jnp.log(len(beta))
    # print(log_likelihoods, log_posterior_primed)
    #implement rejection sampling 
    samples = particles[:,: dimension -1+1]
    # print(samples.shape)
    M = jnp.max( log_likelihoods - log_posterior_primed) 
    print(len(log_posterior_primed))
    u = jax.random.uniform(jax.random.PRNGKey(5), shape=(len(log_posterior_primed),))
    accepted =  log_likelihoods - log_posterior_primed - M > jnp.log(u)

    samples = samples[accepted]
    return samples


def draw_iid_samples(dict):
    result = dict 
    samples         = []
    log_likelihoods = []
    betas           = []
    log_evidences       = []
    log_evidence = 0.0
    for key in result.keys():
    
        samples += list(result[key]['samples'])
        
        log_likelihoods+= list((result[key]['log_likelihoods']))
        betas.append(result[key]['beta'])
         # evidence_piece = np.sum(result[key]['weights'])/len(result[key]['weights'])
        log_evidence_piece = logsumexp(np.log(result[key]['weights'])) - np.log(len(result[key]['weights']))
            
        log_evidence      += log_evidence_piece
        log_evidences.append(log_evidence)
        

    print(log_evidences)
    betas = np.array(betas)
    log_evidences= np.array(log_evidences)
    samples = np.array(samples)

   
   
    log_likelihoods = np.array(log_likelihoods)
    print("length samples: ", samples.shape)
    print("shape log_likelihoods: ", log_likelihoods.shape)
    print(log_likelihoods)
    print("length log_likelihoods: ", log_likelihoods.shape)
    print("betas shape: ", betas.shape)

    log_posterior_primed        = np.array([log_likelihoods * beta - log_evidence for beta, log_evidence in zip(betas, log_evidences)])
    log_posterior_primed        = jnp.logaddexp.reduce( log_posterior_primed, axis = 0) - jnp.log(len(result.keys()))

    print("shape log_posterior_primed: ", log_posterior_primed.shape)

    M = np.max( log_likelihoods - log_posterior_primed)  
    print(len(log_posterior_primed))
    u = np.random.uniform( size = len(log_posterior_primed))
    print("u shape: ", u.shape)
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





        












        

    






    









def compute_evidence(dict):
 
    # with open(result_path, 'r') as f:
    #     result = json.load(f)
    result = dict
    log_evidence = 0.0
    errors    = []
    

    for key in result.keys():
        if float(key) > -1:
            
            # evidence_piece = np.sum(result[key]['weights'])/len(result[key]['weights'])
            log_evidence_piece = logsumexp(np.log(result[key]['weights'])) - np.log(len(result[key]['weights']))
            
            log_evidence      += log_evidence_piece

            ### compute evidence with bootstraping
            log_boot_weights = np.log(result[key]['weights'])
            # evidences = [logsumexp(log_boot_weights[np.random.choice(len(log_boot_weights), len(log_boot_weights))]) -len(log_boot_weights) for _ in range(1000)]

            dlogz_piece = np.var([logsumexp(log_boot_weights[np.random.choice(len(log_boot_weights), len(log_boot_weights))]) -len(log_boot_weights) for _ in range(100)])
            errors.append(dlogz_piece)
            
    
    logz, dlogz  = log_evidence, np.sqrt(np.sum(np.cumsum(errors)))
    # np.savetxt(f'{folder}/{label}/evidence.txt', np.array([logz, dlogz])[np.newaxis], fmt='%3f')

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