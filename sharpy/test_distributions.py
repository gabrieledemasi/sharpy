import jax.numpy as jnp
from jax.scipy.special import logsumexp


def bimodal_gaussian_mixture(mean_1, mean_2, sigma, weight = 0.5, dimensions = 10):
    
    def _bimodal_gaussian_mixture(params):
        mean1   = jnp.ones(dimensions) * mean_1
        mean2   = jnp.ones(dimensions) * mean_2
        
        cov1     = jnp.eye(dimensions) * sigma
        cov2     = jnp.eye(dimensions) * sigma * 4
        inv_cov1 = jnp.linalg.inv(cov1)
        inv_cov2 = jnp.linalg.inv(cov2)
        
        diff1   = params - mean1
        diff2   = params - mean2

        exponent1 = -0.5 * jnp.einsum('...i,ij,...j->...', diff1, inv_cov1, diff1)
        exponent2 = -0.5 * jnp.einsum('...i,ij,...j->...', diff2, inv_cov1, diff2)
        
        norm_const1 = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov1))
        norm_const2 = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov2))
        weight1     = weight
        weight2     = 1 - weight1 

        logpdf1 = exponent1 + norm_const1 + jnp.log(weight1)  
        logpdf2 = exponent2 + norm_const2 + jnp.log(weight2) 

        return logsumexp(jnp.stack([logpdf1, logpdf2]), axis=0) 
        
    return _bimodal_gaussian_mixture



import jax.numpy as jnp






