from jax.scipy.special import ndtri, ndtr
from jax.scipy.stats import norm
import jax.numpy as jnp

def _get_lower_upper(prior_bounds):
    """
    prior_bounds shape: (ndim, 2)
    prior_bounds[:, 0] = lower
    prior_bounds[:, 1] = upper
    """
    prior_bounds = jnp.asarray(prior_bounds)
    lower = prior_bounds[:, 0]
    upper = prior_bounds[:, 1]
    return lower, upper


def from_bounds_to_unit_interval(theta, prior_bounds, eps=None):
    theta = jnp.asarray(theta)
    lower, upper = _get_lower_upper(prior_bounds)

    u = (theta - lower) / (upper - lower)

    if eps is not None:
        u = jnp.clip(u, eps, 1.0 - eps)

    return u


def from_unit_interval_to_bounds(u, prior_bounds):
    u = jnp.asarray(u)
    lower, upper = _get_lower_upper(prior_bounds)

    theta = lower + (upper - lower) * u
    return theta


def from_samples_to_probit(theta, prior_bounds, eps=1e-12):
    """
    theta -> z

    theta is in physical bounded space.
    z is in unconstrained probit space.
    """
    u = from_bounds_to_unit_interval(theta, prior_bounds, eps=eps)
    return ndtri(u)


def from_probit_to_samples(z, prior_bounds):
    """
    z -> theta

    z is unconstrained.
    theta is in physical bounded space.
    """
    z = jnp.asarray(z)
    u = ndtr(z)
    return from_unit_interval_to_bounds(u, prior_bounds)


def log_abs_det_jacobian_probit_to_samples(z, prior_bounds):
    """
    log |d theta / dz|

    theta_i = lower_i + (upper_i - lower_i) Phi(z_i)

    d theta_i / dz_i = (upper_i - lower_i) phi(z_i)
    """
    z = jnp.asarray(z)
    lower, upper = _get_lower_upper(prior_bounds)

    return jnp.sum(
        jnp.log(upper - lower) + norm.logpdf(z),
        axis=-1,
    )


def log_abs_det_jacobian_samples_to_probit(theta, prior_bounds, eps=1e-12):
    """
    log |dz / d theta|
    """
    z = from_samples_to_probit(theta, prior_bounds, eps=eps)
    return -log_abs_det_jacobian_probit_to_samples(z, prior_bounds)