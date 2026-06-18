import jax
from sharpy.utils import vmap_chunked

def build_nuts_kernel_fn(kernel, log_posterior, step_size, max_num_doublings=6):
    

    def _kernel(rng_key, state, beta, metric):
        logdensity_fn = lambda x: log_posterior(x, beta)

        state, info = kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            metric,
            max_num_doublings=max_num_doublings, 
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



def build_hmc_kernel_fn(kernel, log_posterior, ):
    

    def _kernel(rng_key, state,step_size, num_integration_steps, beta, metric):
        logdensity_fn = lambda x: log_posterior(x, beta)

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
            in_axes=(0, 0, 0, 0, 0, 0),
            chunk_size=1000,
            axis_0_is_sharded=False,
        )
    )

    return batched_kernel

