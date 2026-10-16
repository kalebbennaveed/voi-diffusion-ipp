"""Plot the declared environments, distinct from mission outcome evidence."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.scenarios import make_scenario, make_truth

fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
for row, (name, seed) in enumerate([("archipelago", 400), ("channels", 401), ("mosaic", 402)]):
    cfg = load_config(f"configs/persistent_{name}.yaml")
    env, belief, x, q = make_scenario(cfg)
    truth = make_truth(cfg, belief, seed)
    bounds = np.asarray(env.bounds)
    extent = (bounds[0, 0], bounds[1, 0], bounds[0, 1], bounds[1, 1])
    shape = (cfg.grid_ny, cfg.grid_nx)
    targets = np.asarray(env.targets)
    process = np.asarray(env.process_noise)
    wind = np.asarray(truth.at(env.cells))
    fields = [targets, process, np.linalg.norm(wind, axis=1)]
    for col, (field, title, vmax) in enumerate(
        zip(
            fields,
            ["Clarity target", "QoI process noise Qp", "Physical wind (evaluation only)"],
            [1, 0.1, 1.1],
        )
    ):
        ax = axes[row, col]
        img = ax.imshow(
            field.reshape(shape),
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=vmax,
            cmap="viridis",
            interpolation="nearest",
        )
        if col == 2:
            points = np.asarray(env.cells)[::3]
            arrows = wind[::3]
            ax.quiver(
                points[:, 0],
                points[:, 1],
                arrows[:, 0],
                arrows[:, 1],
                color="white",
                scale=8,
                width=0.004,
            )
        for ox, oy, radius in np.asarray(env.obstacles):
            ax.add_patch(Circle((ox, oy), radius, facecolor="#ccd2d5", edgecolor="#344853", lw=1))
        ax.plot(*np.asarray(x[:2]), "o", color="white", mec="#243d46", ms=5)
        ax.set(title=f"{name.title()} · {title}", xlabel="x (m)", ylabel="y (m)")
        fig.colorbar(img, ax=ax, shrink=0.75)
fig.suptitle(
    "Declared persistent-monitoring environments · 475 cells each\nEnvironment definitions, not completed mission results"
)
out = Path("docs/persistent_poam_validation")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "scenarios.png", dpi=160)
print(out / "scenarios.png")
