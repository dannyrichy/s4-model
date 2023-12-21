import jax
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
import jax.numpy as jnp
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve
from functools import partial
import numpy as np


def init_s4_model(rng, N):
    """Initialise Matrices in the state space model
        x' = Ax+Bu
        y = Cx + Du

        Usually D is set as zero.
    Args:
        rng (_type_): _description_
        N (_type_): Dimenstion of x
    """
    # Splitting the rng key into three parts
    a_rng, b_rng, c_rng = jax.random.split(rng, 3)

    a = jax.random.uniform(a_rng, (N, N))
    b = jax.random.uniform(b_rng, (N, 1))
    c = jax.random.uniform(c_rng, (1, N))

    return a, b, c


def cont_2_disc(a, b, step):
    """Function to discretize the transition matrices based on bilinear transformation.
    Omitted transforming C to C_discrete coz it's the same

    Args:
        a (_type_): _description_
        b (_type_): _description_
        step (_type_): _description_

    Returns:
        _type_: _description_
    """
    I = np.eye(a.shape[0])
    tmp = inv(I - (step / 2.0) * a)
    a_approx = tmp @ (I + (step / 2.0) * a)
    b_approx = (tmp * step) @ b
    return a_approx, b_approx


def scan_SSM(a_approx, b_approx, c_approx, u, x0):
    """Utilising scan function from jax to accumulate the values for output

    Args:
        a_approx (_type_): _description_
        b_approx (_type_): _description_
        c_approx (_type_): _description_
        u (_type_): _description_
        x0 (_type_): _description_
    """

    def step(x_k_prev, u_k):
        x_k = a_approx @ x_k_prev + b_approx @ u_k
        y_k = c_approx @ x_k

        # carryover, accumulated
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def run_SSM(a, b, c, u):
    l = u.shape[0]
    n = a.shape[0]
    a_approx, b_approx = cont_2_disc(a=a, b=b, step=1.0 / l)

    # the last index is to get the output
    return scan_SSM(
        a_approx=a_approx,
        b_approx=b_approx,
        c_approx=c,
        u=u[:, jnp.newaxis],
        x0=jnp.zeros((n,)),
    )[1]


# Classical mechanical problem
def example_mass(k, b, m):
    a = np.array([[0, 1], [-k / m, -b / m]])
    b = np.array([[0], [1.0 / m]])
    c = np.array([[1.0, 0]])
    return a, b, c


@partial(jnp.vectorize, signature="()->()")
def example_force(t):
    x = np.sin(10 * t)
    return x * (x > 0.5)

if __name__ == "__main__":
    # Creating a pseudo random number generator key that is passed to most functions
    rng = jax.random.PRNGKey(1)
