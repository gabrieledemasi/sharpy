
import jax
import jax.numpy as jnp



#MASS MATRIX utils

def softabs_lambda(lambdas, alpha):
    """
    Compute the SoftAbs regularized eigenvalues.
    Args:
        lambdas: Eigenvalues of the Hessian.
        alpha: SoftAbs smoothing parameter.
    
    Returns:
        Regularized eigenvalues.
    """
    return lambdas / jnp.tanh(alpha * lambdas)


def softabs_metric(H, alpha = 5e-4):
    """
    Compute the SoftAbs metric tensor given a potential energy function U.
    
    Args:
        U: Potential energy function U(q).
        q: Position variable (state in phase space).
        alpha: SoftAbs regularization parameter (controls smoothness).
    
    Returns:
        SoftAbs metric g(q).
    """

    # Eigen decomposition of the Hessian
    lambdas, V = jnp.linalg.eigh(H)  # H = V D V^T, where D is diagonal of eigenvalues

    # Apply SoftAbs function to eigenvalues
    soft_lambdas = softabs_lambda(lambdas, alpha)

    # Reconstruct metric: g(q) = V Λ_soft V^T
    G = V @ jnp.diag(soft_lambdas) @ V.T

    return G
    

def make_positive_definite(A):
    A = (A + A.T) / 2  # Ensure symmetry
    eigenvalues_, eigenvectors = jnp.linalg.eigh(A)
    
    # Replace non-positive eigenvalues with a small positive number
    eigenvalues = jnp.abs(eigenvalues_)
    
    # Reconstruct the matrix
    A_positive = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
    
    return A_positive


def kinetic_energy(p, inverse_mass_matrix):
    return 0.5*jnp.dot(p.T,jnp.dot(inverse_mass_matrix,p))


def symmetrise(A):
    return (A + A.T) / 2


def compute_mass_matrix(logdensity, q):
    """ see https://arxiv.org/pdf/1212.4693"""

    mass_matrix = -jax.hessian(logdensity)(q)  # Hessian of the log-density

    mass_matrix = softabs_metric(mass_matrix)
    inverse_mass_matrix = jnp.linalg.inv(mass_matrix)
    # logdet = jnp.linalg.slogdet(mass_matrix)[1]
    
    return  inverse_mass_matrix





import jax.numpy as np
import jax
from math import pi, floor


EARTH_SEMI_MAJOR_AXIS = 6378137.0  # for ellipsoid model of Earth, in m
EARTH_SEMI_MINOR_AXIS = 6356752.314  # in m

# Constants
JULIAN_DATE_START_OF_GPS_TIME = 2444244.5
leaps = np.array([
    46828800, 78364801, 109900802, 173059203, 252028804, 315187205, 346723206, 
    393984007, 425520008, 457056009, 504489610, 551750411, 599184012, 820108813, 
    914803214, 1025136015, 1119744016, 1167264017
], dtype=np.float64)

EPOCH_J2000_0_GPS = 630763213

def GreenwichMeanSiderealTime(gpstime) :
    """Calculates Greenwich Mean Sidereal Time given GPS time."""
    return _GreenwichMeanSiderealTime(gpstime)

def _GreenwichMeanSiderealTime(gpstime):
    jd = _GPS2JD(gpstime)
    gps_ns = gpstime - np.round(gpstime)
    t_hi = (jd - 2451545.0) / 36525.0
    t_lo = gps_ns / (36525.0 * 86400.0)
    t = t_hi + t_lo

    sidereal_time = (-6.2e-6 * t + 0.093104) * t * t + 67310.54841
    sidereal_time += 8640184.812866 * t_lo
    sidereal_time += 3155760000.0 * t_lo
    sidereal_time += 8640184.812866 * t_hi
    sidereal_time += 3155760000.0 * t_hi

    return sidereal_time * pi / 43200.0

def GPS2JD(gpstime):
    """Converts GPS time to Julian Date."""
    return _GPS2JD(gpstime)

def _GPS2JD(gpstime):
    """Helper function to compute Julian Date from GPS time."""
    dot2gps = 29224.0
    dot2utc = 2415020.5
    
    # Determine leap seconds
    nleap = jax.lax.cond(
        gpstime < 820108814,  # Condition (must be a JAX expression)
        lambda _: 32,  # If True
        lambda _:     jax.lax.cond(
                                    np.logical_and(gpstime < 914803215,gpstime >820108814),  # Condition (must be a JAX expression)
                                    lambda _: 33,  # If True
                                    lambda _: 34,   # If False
                                    operand=None),
   # If False
        operand=None
    )
#    nleap = jax.lax.cond(
#        820108814 <= gpstime < 914803215,  # Condition (must be a JAX expression)
#        lambda _: 33,  # If True
#        lambda _: 34,   # If False
#        operand=None
#    )

#    if gpstime < 820108814:
#        nleap = 32
#    elif 820108814 <= gpstime < 914803215:
#        nleap = 33
#    else:
#        nleap = 34

    dot = dot2gps + (gpstime - (nleap - 19)) / 86400.0
    utc = dot + dot2utc
    jd = utc

    return jd



def TimeDelayFromEarthCenter( lat, lon, h,  ra,dec,GPS_time,):

    def vertex(lat, lon, h):
        major, minor = EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
        # compute vertex location
        r = major**2 * (
            major**2 * np.cos(lat) ** 2 + minor**2 * np.sin(lat) ** 2
        ) ** (-0.5)
        x = (r + h) * np.cos(lat) * np.cos(lon)
        y = (r + h) * np.cos(lat) * np.sin(lon)
        z = ((minor / major) ** 2 * r + h) * np.sin(lat)
        return np.array([x, y, z])
  
   
    lat = np.radians(lat)
    lon = np.radians(lon)
    delta_d = - vertex(lat, lon, h)
    
    
   
    c  = 2.99792458*1e8
    gmst = GreenwichMeanSiderealTime(GPS_time) 
    
    gmst = np.mod(gmst, 2 * np.pi)
    phi = ra - gmst
    theta = np.pi / 2 - dec
    omega = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )
    return np.dot(omega, delta_d)/c
    





@jax.jit
def McQ2Masses(mc, q):
    """
    | Converts from chirp mass and mass ratio :math:`\\mathcal{M}_c, q` to component masses :math:`m_1, m_2`,
    | with :math:`m_1 \geq m_2` 
    
    :param mc: chirp mass in units of solar masses
    :type mc: float
    :param q: mass ratio
    :type q: float
    
    :return: :math:`m_1, m_2` in units of solar masses
    :rtype: tuple
    """
    
    factor = mc * np.power(1. + q, 1.0/5.0);
    m1     = factor * np.power(q, -3.0/5.0);
    m2     = factor * np.power(q, +2.0/5.0);
    return m1, m2

@jax.jit
def Masses2McQ(m1, m2):
    """
    | Converts from omponent masses :math:`m_1, m_2` (with :math:`m_1 \geq m_2` ) to chirp mass and mass ratio :math:`\\mathcal{M}_c, q` 
    
    :param m1: primary mass in units of solar masses
    :type m1: float
    :param m2: secondary mass in units of solar masses
    :type m2: float
    
    :return: :math:`\\mathcal{M}_c` (in units of solar masses), :math:`q`
    :rtype: tuple
    """
    
    q   = m2/m1
    eta = m1*m2/(m1+m2)
    mc  = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    return mc, q








#################3
# PRIOR UTILS
##################
from jax.scipy.special import logsumexp


def importance_sampling_uniform(
                                key,
                                n_samples,
                                prior_logprob,
                                bounds,):      
    dim = bounds.shape[0]
    
    # Sample from uniform proposal
    x = jax.random.uniform(
        key,
        shape=(n_samples, dim),
        minval=bounds[:, 0],
        maxval=bounds[:, 1],
    )

    # Compute log prior
    logp = jax.vmap(prior_logprob)(x)

    # log q(x) for uniform
    widths = bounds[:, 1] - bounds[:, 0]
    logq = -jnp.sum(jnp.log(widths))

    logw = logp - logq
    logZ = logsumexp(logw) - jnp.log(n_samples)
    # Stabilize weights
    logw = logw - jnp.max(logw)
    w = jnp.exp(logw)
    w = w / jnp.sum(w)

    return x, w, logZ

def sample_from_prior(key, n_particles, prior_logprob, bounds, oversample=5):
    key_prop, key_resample = jax.random.split(key)

    

    while True:
        x, w, logZ = importance_sampling_uniform(
                                                    key_prop,
                                                    n_particles * oversample,
                                                    prior_logprob,
                                                    bounds,
                                                    )
        
        ess = (jnp.sum(w) ** 2) / jnp.sum(w ** 2)
        if ess>= n_particles:
            break
        else:
            oversample *= 2


    initial_evidence = logZ
    idx = jax.random.choice(
                            key_resample,
                            x.shape[0],
                            shape=(n_particles,),
                            p=w,
                            replace=True,
                        )


    return x[idx], initial_evidence







from functools import wraps
from typing import Callable, Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map



from jax.tree_util import tree_leaves, tree_map



def _axis_size(tree, axis: int = 0) -> int:
    """
    Get mapped axis size from the first array leaf of a pytree.
    """
    leaves = tree_leaves(tree)

    if len(leaves) == 0:
        raise ValueError("Cannot infer axis size from an empty pytree.")

    for leaf in leaves:
        if hasattr(leaf, "shape"):
            return leaf.shape[axis]

    raise ValueError("Could not find an array-like leaf with a shape.")


def _check_axis_size(tree, expected_size: int, axis: int = 0):
    """
    Check that every array leaf in a pytree has the expected mapped axis size.
    """
    for leaf in tree_leaves(tree):
        if hasattr(leaf, "shape"):
            if leaf.shape[axis] != expected_size:
                raise ValueError(
                    f"Mapped pytree leaves must all have axis size {expected_size}, "
                    f"but found leaf with shape {leaf.shape}."
                )


def _pad_axis0(x, pad_amount: int):
    """
    Pad a single array on axis 0.
    """
    if pad_amount == 0:
        return x

    pad_width = [(0, 0)] * x.ndim
    pad_width[0] = (0, pad_amount)
    return jnp.pad(x, pad_width)


def _chunk_axis0(x, n_chunks: int, chunk_size: int):
    """
    Reshape array from:

        (padded_n, ...)

    to:

        (n_chunks, chunk_size, ...)
    """
    return x.reshape((n_chunks, chunk_size) + x.shape[1:])


def _unchunk_axis0(x, original_size: int, padded_size: int):
    """
    Reshape array from:

        (n_chunks, chunk_size, ...)

    to:

        (original_size, ...)
    """
    x = x.reshape((padded_size,) + x.shape[2:])
    return x[:original_size]


def vmap_chunked(
    f: Callable,
    in_axes=0,
    *,
    chunk_size: int | None = None,
    axis_0_is_sharded: bool = False,
) -> Callable:
    """
    Chunked version of jax.vmap.

    Supports pytrees as mapped arguments, for example BlackJAX / HMCState objects.

    Example:

        kernel_fn = vmap_chunked(
            kernel_fn_single,
            in_axes=(0, 0, 0, 0),
            chunk_size=1000,
        )

    or:

        batched_single = vmap_chunked(
            single,
            in_axes=(0, None),
            chunk_size=1000,
        )

    Notes:
      - Supports only axis 0 or None in in_axes.
      - Supports pytree inputs and pytree outputs.
      - Pads internally to a multiple of chunk_size, then slices back.
      - axis_0_is_sharded is accepted for API compatibility, but not implemented.
    """

    if chunk_size is None:
        return jax.vmap(f, in_axes=in_axes)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive or None.")

    def normalize_in_axes(num_args: int):
        if isinstance(in_axes, int) or in_axes is None:
            return (in_axes,) * num_args

        if isinstance(in_axes, (tuple, list)):
            if len(in_axes) != num_args:
                raise ValueError(
                    f"in_axes has length {len(in_axes)}, but function got "
                    f"{num_args} positional arguments."
                )
            return tuple(in_axes)

        raise TypeError("Only int, None, tuple, or list in_axes are supported.")

    def check_axes(axes):
        for ax in axes:
            if ax not in (0, None):
                raise NotImplementedError(
                    "This simplified vmap_chunked only supports in_axes entries "
                    "equal to 0 or None."
                )

    @wraps(f)
    def wrapped(*args: Any):
        axes = normalize_in_axes(len(args))
        check_axes(axes)

        mapped_positions = [i for i, ax in enumerate(axes) if ax == 0]

        if len(mapped_positions) == 0:
            return f(*args)

        first_mapped_arg = args[mapped_positions[0]]
        n = _axis_size(first_mapped_arg, axis=0)

        for i in mapped_positions:
            _check_axis_size(args[i], n, axis=0)

        n_chunks = (n + chunk_size - 1) // chunk_size
        padded_n = n_chunks * chunk_size
        pad_amount = padded_n - n

        def pad_and_chunk_tree(tree):
            return tree_map(
                lambda x: _chunk_axis0(_pad_axis0(x, pad_amount), n_chunks, chunk_size)
                if hasattr(x, "shape")
                else x,
                tree,
            )

        chunked_args = {
            i: pad_and_chunk_tree(args[i])
            for i in mapped_positions
        }

        vmapped_f = jax.vmap(f, in_axes=axes)

        def scan_body(_, chunk_index):
            call_args = []

            for i, arg in enumerate(args):
                if axes[i] == 0:
                    chunk_arg = tree_map(
                        lambda x: x[chunk_index] if hasattr(x, "shape") else x,
                        chunked_args[i],
                    )
                    call_args.append(chunk_arg)
                else:
                    call_args.append(arg)

            y = vmapped_f(*call_args)
            return None, y

        _, ys = lax.scan(scan_body, None, jnp.arange(n_chunks))

        def unchunk_tree(tree):
            return tree_map(
                lambda x: _unchunk_axis0(x, n, padded_n)
                if hasattr(x, "shape")
                else x,
                tree,
            )

        return unchunk_tree(ys)

    return wrapped


def local_knn_covariances(
    X,
    weights=None,
    k=1000,
    shrinkage=0.1,
    jitter=1e-5,
    include_self=True,
):
    """
    Estimate one local covariance matrix per particle using k nearest neighbors.

    Parameters
    ----------
    X : array, shape (N, D)
        Particles.

    weights : array, shape (N,), optional
        SMC particle weights. If None, uniform weights are used.

    k : int
        Number of nearest neighbors used for each local covariance.

    shrinkage : float
        Shrinkage toward isotropic covariance.
        Useful range: 0.05 to 0.5.

    jitter : float
        Small diagonal term for numerical stability.

    include_self : bool
        If True, each particle is included among its own neighbors.

    Returns
    -------
    local_means : array, shape (N, D)
        Local weighted mean around each particle.

    local_covs : array, shape (N, D, D)
        Local covariance matrix for each particle.

    local_precisions : array, shape (N, D, D)
        Inverse local covariance matrix for each particle.
    """
    N, D = X.shape

    if weights is None:
        weights = jnp.ones(N) / N
    else:
        weights = weights / jnp.sum(weights)

    # Pairwise squared distances: shape (N, N)
    dists = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)

    if not include_self:
        dists = dists + jnp.eye(N) * 1e30

    # Indices of k nearest neighbors for each particle: shape (N, k)
    knn_idx = jax.lax.top_k(-dists, k)[1]  # (B, k)

    # Neighbor particles: shape (N, k, D)
    X_knn = X[knn_idx]

    # Neighbor weights: shape (N, k)
    w_knn = weights[knn_idx]
    w_knn = w_knn / (jnp.sum(w_knn, axis=1, keepdims=True) + 1e-12)

    # Local weighted mean: shape (N, D)
    local_means = jnp.sum(w_knn[:, :, None] * X_knn, axis=1)

    # Centered local particles: shape (N, k, D)
    diff = X_knn - local_means[:, None, :]

    # Local covariance: shape (N, D, D)
    local_covs = jnp.einsum("nk,nkd,nke->nde", w_knn, diff, diff)

    # Global weighted variance scale for shrinkage target
    global_mean = jnp.sum(weights[:, None] * X, axis=0)
    global_diff = X - global_mean
    global_cov = (weights[:, None] * global_diff).T @ global_diff
    avg_var = jnp.trace(global_cov) / D

    eye = jnp.eye(D)

    # Shrink local covariance toward isotropic covariance
    local_covs = (
        (1.0 - shrinkage) * local_covs
        + shrinkage * avg_var * eye[None, :, :]
    )

    # Add jitter
    local_covs = local_covs + jitter * eye[None, :, :]

    # Local precision matrices
    local_precisions = jnp.linalg.inv(local_covs)

    return local_means, local_covs, local_precisions