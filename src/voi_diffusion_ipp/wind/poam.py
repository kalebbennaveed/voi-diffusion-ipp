"""JAX POAM/Attentive Kernel adapter (Chen et al., RSS 2024).

Implements Sec. III-B AK, Sec. IV-B pivoted inducing selection, Eqs. 19-21
transported UNWEIGHTED sufficient statistics, Eqs. 7-8 variational update and
Eq. 10 minibatch variational EM kernel training. Independently reimplemented
from the paper and MIT-licensed vendor reference; see vendor/POAM-LICENSE.

Full sparse-GP prediction and planning retain the conditional residual covariance.
The continuous posterior sampler is separately validated against this covariance.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .poam_belief import POAMBelief, PosteriorBasis, spectral_nodes


def init_parameters(key, components=4, width=8, prior_std=0.55):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (2, width)) * 0.4,
        "b1": jnp.zeros(width),
        "w2": jax.random.normal(k2, (width, width)) * 0.25,
        "b2": jnp.zeros(width),
        "w3": jax.random.normal(k3, (width, components)) * 0.25,
        "b3": jnp.zeros(components),
        "log_amplitude": jnp.array(np.log(prior_std**2), jnp.float32),
    }


def attention_weights(params, x):
    # This wind model is w(x,y). A timestamp must never become a GP input.
    if x.ndim not in (1, 2) or x.shape[-1] != 2:
        raise ValueError("Spatial wind inputs must contain only (x, y), without time")
    # Optional fixed geometry normalization for the attention network only.
    # Kernel distances and lengthscales remain in physical meters. These two
    # constants are not learned and introduce no temporal coordinate.
    if "attention_offset" in params:
        x = (x-jax.lax.stop_gradient(params["attention_offset"])) / jax.lax.stop_gradient(params["attention_scale"])
    z = jnp.tanh(x @ params["w1"] + params["b1"])
    z = jnp.tanh(z @ params["w2"] + params["b2"])
    w = jax.nn.softmax(z @ params["w3"] + params["b3"], axis=-1)
    return w / jnp.linalg.norm(w, axis=-1, keepdims=True)


def attentive_kernel(params, x, y, lengths):
    wx, wy = attention_weights(params, x), attention_weights(params, y)
    dist = jnp.sum((x[:, None] - y[None]) ** 2, axis=-1)
    base = jnp.exp(-dist[:, :, None] / (2 * lengths**2))
    mixture = jnp.einsum("il,jl,ijl->ij", wx, wy, base)
    return jnp.exp(params["log_amplitude"]) * (wx @ wy.T) * mixture


def chol_solve(A, b):
    L = jnp.linalg.cholesky((A + A.T) * 0.5 + 1e-6 * jnp.eye(A.shape[0]))
    return jsp.linalg.cho_solve((L, True), b)


def transport_statistic_root(root, P, Kuf):
    """Square-root form of B_new = P.T B_old P + Kuf Kuf.T.

    QR compresses the concatenated factor without subtracting positive matrices.
    This is the same full unweighted POAM statistic, not a covariance reset.
    """
    factors = jnp.concatenate((P.T @ root, Kuf), axis=1)
    _, R = jnp.linalg.qr(factors.T, mode="reduced")
    return R.T


def variational_from_root(K, a, root, noise_std):
    """POAM Eqs. 7–8 in whitened square-root algebra, with existing K jitter.

    Forming K+B/sigma² directly can lose positive definiteness in float32 after
    many transports. Whiten the factor and solve I+FF.T/sigma² instead. Computing
    S as V.T V also avoids an asymmetric K @ solve(A,K) covariance product.
    """
    Lk = jnp.linalg.cholesky((K + K.T) / 2 + 1e-6 * jnp.eye(len(K)))
    whiten = lambda y: jsp.linalg.solve_triangular(Lk, y, lower=True)
    F = whiten(root) / noise_std
    C = jnp.eye(len(K)) + F @ F.T
    Lc = jnp.linalg.cholesky((C + C.T) / 2)
    V = jsp.linalg.solve_triangular(Lc, whiten(K), lower=True)
    v = jsp.linalg.solve_triangular(Lc, whiten(a), lower=True) / noise_std**2
    return V.T @ v, V.T @ V


def select_inducing(params, candidates, lengths, count):
    """Pivoted Cholesky; storage and work depend on M+new batch, not history."""
    # JAX's NumPy view is read-only when it already has float64 dtype.
    # Pivoting below modifies the diagonal; own that workspace in both precisions.
    K = np.array(attentive_kernel(params, candidates, candidates, lengths), dtype=np.float64, copy=True)
    # The normalized attentive kernel has a constant diagonal analytically.
    # Restore it before pivoting; otherwise float32 normalization roundoff can
    # choose a different first pivot after an equivalent JIT transformation.
    amplitude = float(np.exp(np.asarray(params["log_amplitude"])))
    np.fill_diagonal(K, amplitude)
    residual = np.diag(K).copy()
    L = np.zeros((len(candidates), count))
    selected = []
    for j in range(count):
        residual[selected] = -np.inf
        tolerance = 8 * np.finfo(np.float32).eps * amplitude
        i = int(np.flatnonzero(residual >= np.max(residual) - tolerance)[0])
        selected.append(i)
        v = max(residual[i], 1e-12)
        L[:, j] = (K[:, i] - L[:, :j] @ L[i, :j]) / np.sqrt(v)
        residual -= L[:, j] ** 2
    return candidates[jnp.array(selected)]


def select_inducing_jax(params, candidates, lengths, count):
    """Same pivoted-Cholesky rule as real POAM, usable inside a hypothetical rollout."""
    K = attentive_kernel(params, candidates, candidates, lengths)
    amplitude = jnp.exp(params["log_amplitude"])
    K = K.at[jnp.diag_indices(len(candidates))].set(amplitude)
    tolerance = 8 * jnp.finfo(jnp.float32).eps * amplitude

    def pivot(j, state):
        residual, L, used, indices = state
        eligible = jnp.where(used, -jnp.inf, residual)
        index = jnp.argmax(eligible >= eligible.max() - tolerance).astype(jnp.int32)
        column = (K[:, index] - L @ L[index]) / jnp.sqrt(jnp.maximum(residual[index], 1e-12))
        return (
            residual - column**2,
            L.at[:, j].set(column),
            used.at[index].set(True),
            indices.at[j].set(index),
        )

    _, _, _, indices = jax.lax.fori_loop(
        0,
        count,
        pivot,
        (
            jnp.full(len(candidates), amplitude, dtype=K.dtype),
            jnp.zeros((len(candidates), count), dtype=K.dtype),
            jnp.zeros(len(candidates), dtype=bool),
            jnp.zeros(count, dtype=jnp.int32),
        ),
    )
    return candidates[indices]


# q(u), locations and accumulated statistics are fixed during the M-step.
# Pass them as dynamic arguments so changing the posterior does not rebuild JIT.
def negative_elbo(params, bx, by, z, mean, covariance, lengths, noise_std, data_count):
    K = attentive_kernel(params, z, z, lengths)
    Kxz = attentive_kernel(params, bx, z, lengths)
    A = chol_solve(K, Kxz.T).T
    mu = A @ jax.lax.stop_gradient(mean)
    S = jax.lax.stop_gradient(covariance)
    var = jnp.exp(params["log_amplitude"]) - jnp.sum(A * Kxz, axis=1) + jnp.sum((A @ S) * A, axis=1)
    ll = -0.5 * jnp.sum((by - mu) ** 2 + var[:, None]) / noise_std**2
    ll *= data_count / len(bx)
    L = jnp.linalg.cholesky(K + 1e-6 * jnp.eye(len(z)))
    LS = jnp.linalg.cholesky(S + 1e-6 * jnp.eye(len(z)))
    kl = 0.5 * (
        2 * jnp.trace(chol_solve(K, S))
        + jnp.sum(mean * chol_solve(K, mean))
        - 2 * len(z)
        + 4 * jnp.sum(jnp.log(jnp.diag(L)) - jnp.log(jnp.diag(LS)))
    )
    return (-ll + kl) / data_count


negative_elbo_gradient = jax.jit(jax.value_and_grad(negative_elbo))


def spatial_observations(positions, measurements):
    """Validate the static 2D vector-field contract, including under JIT.

    Measurement noise is observation error, not temporal wind process noise.
    Shape checks happen at tracing time and add no per-control-step work.
    """
    x, y = jnp.asarray(positions), jnp.asarray(measurements)
    if x.ndim != 2 or x.shape[1] != 2 or y.shape != x.shape:
        raise ValueError("Wind observations require (N, 2) positions and (N, 2) velocities; no time column")
    return x, y


class POAM:
    """Posterior over a fixed spatial wind field; updated by observations only.

    No time coordinate, forgetting, temporal kernel, or wind process noise.
    Kernel adaptation only between plans; spatially nonstationary AK weights
    do not imply temporal variation of the underlying field.
    """

    @classmethod
    def from_data(cls, positions, measurements, key, inducing_count, train_steps=10, **kwargs):
        """Initialize inducing locations from available data, then run ordinary POAM.

        The reference POAM initializes Z from training locations. A coverage grid
        containing unobserved locations can instead spend most of a small sparse
        model's capacity away from its data. Pivoting the initial data makes this
        choice deterministic; no physical field or unobserved value is accessed.
        This constructor requires at least M distinct initial locations.
        """
        # Do not round actual input locations when inference runs in float64.
        x = jnp.asarray(positions)
        if len(np.unique(np.asarray(x), axis=0)) < inducing_count:
            raise ValueError("Data-supported POAM initialization needs M distinct locations")
        model = cls(x[:inducing_count], key, **kwargs)
        z = select_inducing(model.parameters, x, model.lengths, inducing_count)
        model.z = z
        model.old_K = model.kernel(z, z)
        model.covariance = model.old_K
        model.update(x, measurements, key, train_steps=train_steps, relocate=False)
        return model

    def __init__(
        self,
        centers,
        key,
        noise_std=0.1,
        lengths=(0.3, 0.8, 1.3, 1.8),
        prior_std=0.55,
        spectral_count=32,
        prior_support=None,
        attention_bounds=None,
    ):
        if noise_std <= 0 or prior_std <= 0:
            raise ValueError(
                "Positive observation noise and prior wind standard deviation required"
            )
        self.z = jnp.asarray(centers, jnp.float32)
        self.parameters = init_parameters(key, len(lengths), prior_std=prior_std)
        if attention_bounds is not None:
            bounds = np.asarray(attention_bounds)
            if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)) or not np.all(bounds[1] > bounds[0]):
                raise ValueError("Attention bounds require finite lower/upper spatial corners")
            self.parameters["attention_offset"] = jnp.asarray(bounds.mean(axis=0))
            self.parameters["attention_scale"] = jnp.asarray((bounds[1]-bounds[0])/2)
        self.lengths = jnp.asarray(lengths, jnp.float32)
        self.noise_std = float(noise_std)
        self.frequencies = spectral_nodes(spectral_count)
        self.prior_support = (
            jnp.zeros((0, 2)) if prior_support is None else jnp.asarray(prior_support)
        )
        self.adam_m = jax.tree.map(jnp.zeros_like, self.parameters)
        self.adam_v = jax.tree.map(jnp.zeros_like, self.parameters)
        self.adam_step = 0
        n = len(centers)
        self.a = jnp.zeros((n, 2))
        self.B = jnp.zeros((n, n))
        self.root_B = jnp.zeros((n, n))
        self.old_K = self.kernel(self.z, self.z)
        self.mean = jnp.zeros((n, 2))
        self.covariance = self.old_K
        self.data_x = jnp.zeros((0, 2))
        self.data_y = jnp.zeros((0, 2))

    def kernel(self, x, y):
        return attentive_kernel(self.parameters, x, y, self.lengths)

    def update(
        self, positions, measurements, key, train_steps=4, batch_size=32, relocate=True, lr=0.001
    ):
        x, y = spatial_observations(positions, measurements)
        old_z = self.z
        z = (
            select_inducing(self.parameters, jnp.concatenate((old_z, x)), self.lengths, len(old_z))
            if relocate
            else old_z
        )
        P = chol_solve(self.old_K, self.kernel(old_z, z))
        K = self.kernel(z, z)
        Kuf = self.kernel(z, x)
        self.a = P.T @ self.a + Kuf @ y
        self.root_B = transport_statistic_root(self.root_B, P, Kuf)
        self.B = self.root_B @ self.root_B.T
        self.mean, self.covariance = variational_from_root(K, self.a, self.root_B, self.noise_std)
        self.z = z
        self.old_K = K
        self.data_x = jnp.concatenate((self.data_x, x))
        self.data_y = jnp.concatenate((self.data_y, y))

        # Same variational EM objective; reusable compiled gradient.
        losses = []
        m, v = self.adam_m, self.adam_v
        for i in range(train_steps):
            self.adam_step += 1
            key, k = jax.random.split(key)
            idx = jax.random.randint(k, (min(batch_size, len(self.data_x)),), 0, len(self.data_x))
            loss, g = negative_elbo_gradient(
                self.parameters,
                self.data_x[idx],
                self.data_y[idx],
                z,
                self.mean,
                self.covariance,
                self.lengths,
                jnp.array(self.noise_std),
                jnp.array(len(self.data_x), dtype=jnp.float32),
            )
            losses.append(float(loss))
            m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
            v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
            self.parameters = jax.tree.map(
                lambda p, mm, vv: (
                    p
                    - lr
                    * (mm / (1 - 0.9**self.adam_step))
                    / (jnp.sqrt(vv / (1 - 0.999**self.adam_step)) + 1e-8)
                ),
                self.parameters,
                m,
                v,
            )
        self.adam_m, self.adam_v = m, v
        return losses

    def joint(self, locations):
        x = jnp.atleast_2d(locations)
        Kxz = self.kernel(x, self.z)
        K = self.kernel(self.z, self.z)
        A = chol_solve(K, Kxz.T).T
        cov = self.kernel(x, x) - A @ Kxz.T + A @ self.covariance @ A.T
        return A @ self.mean, (cov + cov.T) * 0.5

    def predict(self, locations):
        mu, cov = self.joint(locations)
        return mu, jnp.maximum(jnp.diag(cov), 0)[:, None, None] * jnp.eye(2)[None]

    def planning_belief(self, anchors=None):
        """Immutable FULL predictive model; hypothetical updates keep kernel/Z fixed.

        No interpolation anchors, no discarded conditional residual. The argument
        is accepted only for old callers passing the current inducing locations.
        """
        if anchors is not None and not np.array_equal(np.asarray(anchors), np.asarray(self.z)):
            raise ValueError("POAM planning uses its full model, not an anchor projection")
        K = self.kernel(self.z, self.z)
        support = self.prior_support
        Ks = self.kernel(support, support)
        regularized = (Ks + Ks.T) * 0.5 + 1e-4 * jnp.eye(len(support))
        Ls = jnp.linalg.cholesky(regularized)
        inverse_s = jsp.linalg.cho_solve((Ls, True), jnp.eye(len(support)))
        basis = PosteriorBasis(
            self.z,
            self.parameters,
            self.lengths,
            self.frequencies,
            support,
            inverse_s,
            Ls,
            attention_weights(self.parameters, support),
            attention_weights(self.parameters, self.z),
        )
        return POAMBelief(
            basis,
            self.mean,
            self.covariance,
            chol_solve(K, jnp.eye(len(self.z))),
            self.a,
            self.B,
            self.old_K,
            jnp.array(self.noise_std),
            self.root_B,
        )
