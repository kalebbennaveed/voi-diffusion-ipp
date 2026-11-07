"""Inference checks for actual-observation-supported POAM allocation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/airflow"))

import jax
import jax.numpy as jnp
import numpy as np

from observed_poam import condition_observed
from supported_poam import condition_supported, capacity_saturated
from voi_diffusion_ipp.wind.poam import POAM


def fixture():
    with jax.experimental.enable_x64():
        xx, yy = jnp.meshgrid(jnp.linspace(0., 2., 4), jnp.linspace(0., 2., 4))
        centers = jnp.stack((xx.ravel(), yy.ravel()), axis=1)
        return POAM(centers, jax.random.PRNGKey(71), prior_std=1.8).planning_belief()


def test_existing_center_matches_original_statistic_update():
    with jax.experimental.enable_x64():
        b = jax.tree.map(lambda x: x.astype(jnp.float64) if hasattr(x, "dtype") and
                        jnp.issubdtype(x.dtype, jnp.floating) else x, fixture())
        p = b.basis.centers[2:3]
        y = jnp.array([[.8, -.2]], dtype=jnp.float64)
        expected = b.condition(p, y, relocate=False)
        got = condition_observed(b, p, y)
        for a, c in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
            np.testing.assert_allclose(a, c, rtol=2e-10, atol=2e-10)


def test_measurement_allocates_support_and_remains_finite_after_repeats():
    with jax.experimental.enable_x64():
        b = fixture()
        p = jnp.array([[.30, .30]], dtype=jnp.float64)
        before = float(b.predict(p)[1][0, 0, 0])
        y = jnp.array([[.8, -.2]], dtype=jnp.float64)
        first = condition_observed(b, p, y)
        assert len(first.mean) == len(b.mean)
        assert np.min(np.linalg.norm(np.asarray(first.basis.centers)-p, axis=1)) < 1e-12
        after = first
        for _ in range(24):
            after = condition_observed(after, p, y)
        np.testing.assert_array_equal(after.basis.centers, first.basis.centers)
        assert all(np.all(np.isfinite(a)) for a in jax.tree.leaves(after))
        variance = float(after.predict(p)[1][0, 0, 0])
        assert 0 < variance < .001
        assert variance < before / 100
        # The conditional residual is preserved away from measured support.
        distant = float(after.predict(jnp.array([[3., 3.]]))[1][0, 0, 0])
        assert distant > variance * 100


def test_supported_existing_center_matches_original_update():
    with jax.experimental.enable_x64():
        b = jax.tree.map(lambda x: x.astype(jnp.float64) if hasattr(x, "dtype") and
                        jnp.issubdtype(x.dtype, jnp.floating) else x, fixture())
        p = b.basis.centers[2:3]
        y = jnp.array([[.8, -.2]], dtype=jnp.float64)
        expected = b.condition(p, y, relocate=False)
        got, mask = condition_supported(b, p, y, jnp.zeros(len(b.mean), dtype=bool))
        assert int(mask.sum()) == 1 and bool(mask[2])
        for a, c in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
            np.testing.assert_allclose(a, c, rtol=2e-10, atol=2e-10)


def test_supported_memory_retained_until_declared_capacity():
    with jax.experimental.enable_x64():
        b = fixture()
        mask = jnp.zeros(len(b.mean), dtype=bool)
        y = jnp.array([[.8, -.2]], dtype=jnp.float64)
        recorded = []
        for point in [[.3, .3], [.5, .3], [.7, .3]]:
            b, mask = condition_supported(b, jnp.array([point]), y, mask)
            recorded.append(point)
            assert all(np.min(np.linalg.norm(np.asarray(b.basis.centers)-p, axis=1)) < 1e-12
                       for p in recorded)
        assert int(mask.sum()) == 3
        assert not bool(capacity_saturated(mask))
        old_centers = np.asarray(b.basis.centers).copy()
        b, mask = condition_supported(b, jnp.array([[.72, .3]]), y, mask)
        np.testing.assert_array_equal(b.basis.centers, old_centers)
        assert all(np.all(np.isfinite(a)) for a in jax.tree.leaves(b))
        # Once capacity is explicitly exhausted, the declared nearest-center
        # replacement remains finite and the mask continues reporting saturation.
        mask = jnp.ones(len(b.mean), dtype=bool)
        b, mask = condition_supported(b, jnp.array([[2.3, 2.3]]), y, mask)
        assert bool(capacity_saturated(mask))
        assert np.min(np.linalg.norm(np.asarray(b.basis.centers)-[2.3, 2.3], axis=1)) < 1e-12
        assert all(np.all(np.isfinite(a)) for a in jax.tree.leaves(b))
