"""Independent flexible speed-profile check for the known-wind route audit."""

import json
import time
from pathlib import Path

import numpy as np
import yaml
from audit_two_region_physics import clearance, field, frame
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]


def geometry(s, spec, offset, reverse):
    a, b, _, n = frame(spec)
    r = 1 - s if reverse else s
    bump = 16 * r**2 * (1 - r) ** 2
    db = 32 * r - 96 * r**2 + 64 * r**3
    ddb = 32 - 192 * r + 192 * r**2
    p = a + r[:, None] * (b - a) + offset * bump[:, None] * n
    ps = ((b - a) + offset * db[:, None] * n) * (-1 if reverse else 1)
    pss = offset * ddb[:, None] * n
    return p, ps, pss


def solve_speed(spec, protocol, offset, reverse, intervals, drag=0.6, amplitude=1.8):
    # Node speed squared y; endpoints are fixed to zero, rest-to-rest.
    N = intervals
    nodes = np.linspace(0, 1, N + 1)
    sample = (np.arange(N)[:, None] + np.linspace(0, 1, 5)[None]).ravel() / N
    idx = np.repeat(np.arange(N), 5)
    fraction = np.tile(np.linspace(0, 1, 5), N)
    W = np.zeros((len(sample), N + 1))
    W[np.arange(len(sample)), idx] = 1 - fraction
    W[np.arange(len(sample)), idx + 1] = fraction
    D = np.zeros_like(W)
    D[np.arange(len(sample)), idx] = -N
    D[np.arange(len(sample)), idx + 1] = N
    W, D = W[:, 1:-1], D[:, 1:-1]
    p, ps, pss = geometry(sample, spec, offset, reverse)
    wind = field(p, spec, protocol, amplitude)
    linear = pss[:, :, None] * W[:, None, :] + 0.5 * ps[:, :, None] * D[:, None, :]
    limit = spec["maximum_control_acceleration"] - 0.015

    def objective(y):
        root = np.sqrt(np.r_[0, y, 0])
        denominator = root[:-1] + root[1:]
        value = np.sum(2 / N / denominator)
        grad = -(1 / N) * (1 / denominator[:-1] ** 2 + 1 / denominator[1:] ** 2) / root[1:-1]
        return value, grad

    def controls(y):
        squared = W @ y
        root = np.sqrt(np.maximum(squared, 0))
        return np.einsum("icj,j->ic", linear, y) + drag * ps * root[:, None] - wind

    def constraint(y):
        u = controls(y).ravel()
        return np.r_[limit - u, limit + u]

    def jacobian(y):
        squared = W @ y
        inverse = np.divide(
            0.5, np.sqrt(np.maximum(squared, 0)), out=np.zeros_like(squared), where=squared > 1e-12
        )
        j = linear + drag * ps[:, :, None] * inverse[:, None, None] * W[:, None, :]
        j = j.reshape(-1, N - 1)
        return np.concatenate((-j, j))

    start = time.perf_counter()
    result = minimize(
        objective,
        np.full(N - 1, 0.03),
        jac=True,
        method="SLSQP",
        bounds=[(1e-7, 4.0)] * (N - 1),
        constraints={"type": "ineq", "fun": constraint, "jac": jacobian},
        options={"maxiter": 400, "ftol": 1e-9},
    )
    residual = float(max(0, -constraint(result.x).min()))
    s = np.linspace(0, 1, 20001)
    k = np.minimum((s * N).astype(int), N - 1)
    w = np.r_[0, result.x, 0]
    speed2 = np.interp(s, nodes, w)
    acceleration = np.diff(w)[k] * N / 2
    p, ps, pss = geometry(s, spec, offset, reverse)
    u = (
        pss * speed2[:, None]
        + ps * acceleration[:, None]
        + drag * ps * np.sqrt(speed2[:, None])
        - field(p, spec, protocol, amplitude)
    )
    return {
        "drag": drag,
        "amplitude": amplitude,
        "offset": offset,
        "direction": "B_to_A" if reverse else "A_to_B",
        "intervals": N,
        "seconds": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": result.nit,
        "wall_seconds": time.perf_counter() - start,
        "constraint_residual": residual,
        "dense_peak_control": float(np.abs(u).max()),
        "clearance": clearance(p, spec),
        "squared_speeds": w.tolist(),
    }


def main():
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    protocol = json.loads((ROOT / "docs/two_region_physics/protocol.json").read_text())
    records = []
    for drag in [0.0, 0.6, 1.0]:
        for amplitude in [0.0, 1.2, 1.5, 1.8]:
            for N in [40, 80]:
                for offset in [-1.5, 0.0, 1.5]:
                    for reverse in [False, True]:
                        row = solve_speed(spec, protocol, offset, reverse, N, drag, amplitude)
                        records.append(row)
                        print(
                            json.dumps({k: v for k, v in row.items() if k != "squared_speeds"}),
                            flush=True,
                        )
                        (ROOT / "docs/two_region_physics/flexible_screen.json").write_text(
                            json.dumps(records, indent=2) + "\n"
                        )


if __name__ == "__main__":
    main()
