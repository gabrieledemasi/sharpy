import jax
from utils import compute_mass_matrix
import jax.numpy as jnp 


def build_mass_matrix_fn(log_posterior):
    def single(pos, beta):
        logdensity = lambda x: log_posterior(x, beta)
        
        return compute_mass_matrix(logdensity, pos)
    return jax.jit(jax.vmap(single, in_axes=(0, None)))



def make_smc_step_fn(init_fn, kernel_fn,log_posterior):
    def make_a_step_vectorized(position, keys, beta ,matrices):
        """
        position: (M, D)
        beta: scalar or vector
        """
        logdensity_fn = lambda x: log_posterior(x, beta)  # Only for init, not passed into JIT

        # Initialize state
        state = init_fn(position, logdensity_fn,)

        beta_batch = jnp.broadcast_to(beta, (position.shape[0],))

        state, info = kernel_fn(keys, state,  beta_batch, matrices)

        return state.position
    
    return make_a_step_vectorized


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
        
        log_weights = log_weights - jnp.max(log_weights)
        weights = jnp.exp(log_weights)
        weights = weights / jnp.sum(weights)
        ess = (jnp.sum(weights)) ** 2 / jnp.sum(weights**2)

        return weights, ess
    return compute_weight_and_ess