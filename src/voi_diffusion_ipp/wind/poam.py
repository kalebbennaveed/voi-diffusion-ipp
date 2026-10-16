"""JAX POAM/Attentive Kernel adapter (Chen et al., RSS 2024).

Implements Sec. III-B AK, Sec. IV-B pivoted inducing selection, Eqs. 19-21
transported UNWEIGHTED sufficient statistics, Eqs. 7-8 variational update and
Eq. 10 minibatch variational EM kernel training. Independently reimplemented
from the paper and MIT-licensed vendor reference; see docs/reference_mapping.md.

Full sparse-GP prediction includes conditional residual covariance. Planning uses
an explicit finite-anchor snapshot: full covariance at anchors, Nyström extension
between anchors (residual omitted there). This approximation is NOT the full GP.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .gp import WindBelief


def init_parameters(key, components=4, width=8):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (2, width)) * 0.4,
        "b1": jnp.zeros(width),
        "w2": jax.random.normal(k2, (width, width)) * 0.25,
        "b2": jnp.zeros(width),
        "w3": jax.random.normal(k3, (width, components)) * 0.25,
        "b3": jnp.zeros(components),
        "log_amplitude": jnp.array(np.log(0.55**2), jnp.float32),
    }


def attentive_kernel(params, x, y, lengths):
    def features(x):
        z = jnp.tanh(x @ params["w1"] + params["b1"])
        z = jnp.tanh(z @ params["w2"] + params["b2"])
        w = jax.nn.softmax(z @ params["w3"] + params["b3"], axis=-1)
        return w / jnp.linalg.norm(w, axis=-1, keepdims=True)

    wx, wy = features(x), features(y)
    dist = jnp.sum((x[:, None] - y[None]) ** 2, axis=-1)
    base = jnp.exp(-dist[:, :, None] / (2 * lengths**2))
    mixture = jnp.einsum("il,jl,ijl->ij", wx, wy, base)
    return jnp.exp(params["log_amplitude"]) * (wx @ wy.T) * mixture


def chol_solve(A, b):
    L = jnp.linalg.cholesky((A + A.T) * 0.5 + 1e-6 * jnp.eye(A.shape[0]))
    return jsp.linalg.cho_solve((L, True), b)


def select_inducing(params, candidates, lengths, count):
    """Pivoted Cholesky; storage and work depend on M+new batch, not history."""
    K = np.asarray(attentive_kernel(params, candidates, candidates, lengths), dtype=np.float64)
    residual = np.diag(K).copy()
    L = np.zeros((len(candidates), count))
    selected = []
    for j in range(count):
        residual[selected] = -np.inf
        i = int(np.argmax(residual))
        selected.append(i)
        v = max(residual[i], 1e-12)
        L[:, j] = (K[:, i] - L[:, :j] @ L[i, :j]) / np.sqrt(v)
        residual -= L[:, j] ** 2
    return candidates[jnp.array(selected)]


class KernelBasis(NamedTuple):
    centers: jax.Array
    parameters: dict
    lengths: jax.Array
    inverse: jax.Array

    def features(self, locations):
        return (
            attentive_kernel(self.parameters, jnp.atleast_2d(locations), self.centers, self.lengths)
            @ self.inverse
        )


class POAM:
    """Online adapter with full prediction. Kernel adaptation only between plans."""

    def __init__(self, centers, key, noise_std=0.1, lengths=(0.25, 0.6, 1.1, 1.8)):
        self.z = jnp.asarray(centers, jnp.float32)
        self.parameters = init_parameters(key, len(lengths))
        self.lengths = jnp.asarray(lengths, jnp.float32)
        self.noise_std = float(noise_std)
        n = len(centers)
        self.a = jnp.zeros((n, 2))
        self.B = jnp.zeros((n, n))
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
        x = jnp.asarray(positions)
        y = jnp.asarray(measurements)
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
        self.B = P.T @ self.B @ P + Kuf @ Kuf.T
        A = K + self.B / self.noise_std**2
        self.mean = K @ chol_solve(A, self.a) / self.noise_std**2
        self.covariance = K @ chol_solve(A, K)
        self.covariance = (self.covariance + self.covariance.T) * 0.5
        self.z = z
        self.old_K = K
        self.data_x = jnp.concatenate((self.data_x, x))
        self.data_y = jnp.concatenate((self.data_y, y))

        # Variational EM M-step: q(u), locations and accumulated statistics fixed.
        def negative_elbo(params, bx, by):
            K = attentive_kernel(params, z, z, self.lengths)
            Kxz = attentive_kernel(params, bx, z, self.lengths)
            A = chol_solve(K, Kxz.T).T
            mu = A @ jax.lax.stop_gradient(self.mean)
            S = jax.lax.stop_gradient(self.covariance)
            var = (
                jnp.exp(params["log_amplitude"])
                - jnp.sum(A * Kxz, axis=1)
                + jnp.sum((A @ S) * A, axis=1)
            )
            ll = -0.5 * jnp.sum((by - mu) ** 2 + var[:, None]) / self.noise_std**2
            ll *= len(self.data_x) / len(bx)
            L = jnp.linalg.cholesky(K + 1e-6 * jnp.eye(len(z)))
            LS = jnp.linalg.cholesky(S + 1e-6 * jnp.eye(len(z)))
            kl = 0.5 * (
                2 * jnp.trace(chol_solve(K, S))
                + jnp.sum(self.mean * chol_solve(K, self.mean))
                - 2 * len(z)
                + 4 * jnp.sum(jnp.log(jnp.diag(L)) - jnp.log(jnp.diag(LS)))
            )
            return (-ll + kl) / len(self.data_x)

        grad = jax.jit(jax.value_and_grad(negative_elbo))
        losses = []
        m = jax.tree.map(jnp.zeros_like, self.parameters)
        v = jax.tree.map(jnp.zeros_like, self.parameters)
        for i in range(train_steps):
            key, k = jax.random.split(key)
            idx = jax.random.randint(k, (min(batch_size, len(self.data_x)),), 0, len(self.data_x))
            loss, g = grad(self.parameters, self.data_x[idx], self.data_y[idx])
            losses.append(float(loss))
            m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
            v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
            self.parameters = jax.tree.map(
                lambda p, mm, vv, i=i: (
                    p
                    - lr
                    * (mm / (1 - 0.9 ** (i + 1)))
                    / (jnp.sqrt(vv / (1 - 0.999 ** (i + 1))) + 1e-8)
                ),
                self.parameters,
                m,
                v,
            )
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
        anchors = self.z if anchors is None else jnp.asarray(anchors)
        mu, cov = self.joint(anchors)
        K = self.kernel(anchors, anchors)
        basis = KernelBasis(
            anchors, self.parameters, self.lengths, chol_solve(K, jnp.eye(len(anchors)))
        )
        return WindBelief(basis, mu, cov, jnp.array(self.noise_std))

    def snapshot_residual_variance(self, locations, anchors=None):
        snap = self.planning_belief(anchors)
        return jnp.diag(self.joint(locations)[1] - snap.joint(locations)[1])
