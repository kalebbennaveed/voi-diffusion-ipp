"""Prospective mission-duration check on fresh paired wind seeds, same planner budget."""

from dataclasses import replace

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.run import mission

if __name__ == "__main__":
    cfg = load_config("configs/long_mission.yaml")
    for method in cfg.methods:
        for seed in cfg.seeds:
            mission(cfg, method, seed, "results/long_mission")
    for seed in cfg.seeds:
        mission(replace(cfg, measurements=False), "proposed", seed, "results/long_no_measurements")
