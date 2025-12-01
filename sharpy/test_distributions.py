import jax.numpy as jnp
from jax.scipy.special import logsumexp


def bimodal_gaussian_mixture(mean_1, mean_2, sigma, weight = 0.5, dimensions = 10):
    
    def _bimodal_gaussian_mixture(params):
        mean1   = jnp.ones(dimensions) * mean_1
        mean2   = jnp.ones(dimensions) * mean_2
        
        cov     = jnp.eye(dimensions) * sigma
        inv_cov = jnp.linalg.inv(cov)
        
        diff1   = params - mean1
        diff2   = params - mean2

        exponent1 = -0.5 * jnp.einsum('...i,ij,...j->...', diff1, inv_cov, diff1)
        exponent2 = -0.5 * jnp.einsum('...i,ij,...j->...', diff2, inv_cov, diff2)
        
        norm_const = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov))
        weight1     = weight
        weight2     = 1 - weight1 

        logpdf1 = exponent1 + norm_const + jnp.log(weight1)  
        logpdf2 = exponent2 + norm_const + jnp.log(weight2) 

        return logsumexp(jnp.stack([logpdf1, logpdf2]), axis=0) 
        
    return _bimodal_gaussian_mixture



import jax.numpy as jnp



def eggbox(params):
    

    x = params
    def _eggbox(x):
        
        return (2.0 + jnp.cos(x[0]/2) * jnp.cos(x[1]/2)) ** 5.0
    return _eggbox(x)






