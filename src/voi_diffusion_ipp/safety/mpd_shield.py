"""Independent JAX implementation of Safe-MPD's predictive backup switch.

Kim et al., arXiv:2512.06261, Algorithm 2. Adaptation to uncertain spatial wind:
accept only if all belief-derived inner fields pass a proposed-step/backup
check. Never use an outer simulated or physical field to choose the action.
Terminal low speed is an empirical recovery condition, not a certified robust
invariant set. The paper's deterministic guarantee therefore does not apply.
No barrier differentiation or QP is used. No reference-repository imports.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..dynamics import step
from ..wind import InnerFields
from .rpcbf import design_policy


class ShieldInfo(NamedTuple):
    active: jax.Array
    accepted: jax.Array
    maximum_violation: jax.Array
    terminal_speed: jax.Array
    correction: jax.Array


def monitor(x, nominal, belief, inner, env, cfg):
    """Proposed step then backup, including intermediate physical integration.

    Feedback uses the common earned belief at each hypothetical state; the
    sampled field supplies dynamics only. Fixed shapes permit jit/vmap/scan.
    """
    if not isinstance(inner, InnerFields):
        raise TypeError("Shield requires belief-derived InnerFields, never WorldField")

    def one(coeff):
        wind = lambda p: inner.basis.evaluate(p, coeff)[0]

        def integrate(z, u, dt):
            def tick(carry, _):
                state, violation = carry
                state = step(state, u, wind, dt / cfg.shield_substeps, cfg.linear_drag)
                violation = jnp.maximum(violation, env.violations(state, cfg.safety_margin).max())
                return (state, violation), None

            initial = env.violations(z, cfg.safety_margin).max()
            return jax.lax.scan(tick, (z, initial), None, length=cfg.shield_substeps)[0]

        proposed, violation = integrate(x, nominal, cfg.dt)

        def backup(carry, _):
            state, maximum = carry
            u = design_policy(state, belief, cfg)
            state, v = integrate(state, u, cfg.backup_dt)
            return (state, jnp.maximum(maximum, v)), None

        (terminal, maximum), _ = jax.lax.scan(
            backup, (proposed, violation), None, length=cfg.backup_steps
        )
        return maximum, jnp.linalg.norm(terminal[2:])

    violations, speeds = jax.vmap(one)(inner.coefficients)
    vmax, speed = violations.max(), speeds.max()
    accepted = jnp.isfinite(vmax) & jnp.isfinite(speed) & (vmax <= 0)
    accepted &= speed <= cfg.shield_terminal_speed
    return accepted, vmax, speed


def filter_control(x, nominal, belief, inner, env, cfg, active=False):
    """Paper latch by default; optional reference-style per-step reconsideration."""
    if not isinstance(inner, InnerFields):
        raise TypeError("Shield requires belief-derived InnerFields, never WorldField")
    if not cfg.shield_latch:
        active = jnp.array(False)
    accepted, violation, speed = jax.lax.cond(
        active,
        lambda _: (jnp.array(False), jnp.array(0.0), jnp.array(0.0)),
        lambda _: monitor(x, nominal, belief, inner, env, cfg),
        operand=None,
    )
    active = active | ~accepted
    control = jnp.where(active, design_policy(x, belief, cfg), nominal)
    return control, ShieldInfo(
        active, accepted, violation, speed, jnp.linalg.norm(control - nominal)
    )
