from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    scenario: str = "corridor"
    dt: float = 0.15
    stage_steps: int = 12
    future_steps: int = 32
    mission_horizons: int = 5
    knots: int = 3
    branches: int = 2
    worlds: int = 4
    inner_samples: int = 6
    backup_steps: int = 16
    backup_dt: float = 0.12
    backup_gain: float = 2.0
    cbf_alpha: float = 1.0
    umax: float = 1.6
    linear_drag: float = 0.0  # s^-1; wind remains acceleration d=linear_drag*w_air
    safety_margin: float = 0.06
    diffusion_steps: int = 6
    proposals: int = 16
    temperature: float = 0.035
    beta_min: float = 0.005
    beta_max: float = 0.18
    candidate_chunk: int = 2
    terminal_seconds: float = 0.0
    sensor_sigma: float = 0.38
    sensor_R: float = 0.18
    clarity_decay: float = 0.035
    clarity_decay_pattern: str = "uniform"
    initial_clarity: str = "cold"
    wind_std: float = 0.55
    wind_length: float = 1.1
    observation_std: float = 0.10
    ig_weight: float = 0.025
    measurements: bool = True
    shared_future: bool = False
    independent_cells: bool = False
    spline: str = "full"
    safety_filter: str = "rpcbf"  # experimental alternative: sampled_mpd_shield
    shield_terminal_speed: float = 0.2  # empirical recovery test, NOT invariant-set proof
    shield_substeps: int = 4
    wind_backend: str = "feature"
    poam_train_steps: int = 10
    hypothetical_relocation: bool = False
    budgeted_policy: str = "compact"  # compact continuous shift or posterior-gated branches
    budgeted_update_steps: int = 0  # 0: one future stage; >0: causal future belief periods
    poam_inducing_x: int = 6
    poam_inducing_y: int = 4
    poam_spectral_nodes: int = 32
    poam_support_x: int = 0
    poam_support_y: int = 0
    grid_nx: int = 25
    grid_ny: int = 19
    whole_field: bool = False
    reject_collisions: bool = True
    warm_starts: int = 1
    obstacle_radius: float = 0.79
    control_representation: str = "acceleration"
    branch_rule: str = "crosswind"
    recourse_samples: int = 3
    waypoint_x_limit: float = 2.25
    waypoint_y_limit: float = 1.65
    future_updates: bool = False
    require_feasible_candidates: bool = False
    seeds: tuple = (0, 1, 2)
    methods: tuple = ("proposed", "passive", "weighted_ig", "heuristic", "cem", "mppi", "oracle")

    def __post_init__(self):
        if self.safety_filter not in ("rpcbf", "sampled_mpd_shield"):
            raise ValueError("Unknown safety filter")
        if self.shield_terminal_speed <= 0 or self.shield_substeps < 1:
            raise ValueError("Positive shield terminal speed and substeps required")
        if self.budgeted_policy not in ("compact", "branches"):
            raise ValueError("Unknown budgeted feedback policy")
        if self.budgeted_update_steps < 0 or (
            self.budgeted_update_steps and self.future_steps % self.budgeted_update_steps
        ):
            raise ValueError("Budgeted update interval must divide the future horizon")
        if not 0 <= self.linear_drag < float("inf"):
            raise ValueError("Linear drag must be finite and nonnegative")
        for name in (
            "stage_steps",
            "future_steps",
            "knots",
            "branches",
            "worlds",
            "inner_samples",
            "backup_steps",
            "diffusion_steps",
            "proposals",
            "candidate_chunk",
            "mission_horizons",
            "recourse_samples",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.backup_steps < 3 or self.knots < 2:
            raise ValueError("Cubic spline requires >=3 backup steps; interpolation >=2 knots")
        if not 0 < self.beta_min <= self.beta_max < 1 or self.temperature <= 0:
            raise ValueError("Invalid diffusion schedule/temperature")
        if self.observation_std <= 0 or self.dt <= 0:
            raise ValueError("Positive observation noise and timestep required")
        if self.branches not in (1, 2):
            raise ValueError("Current observation router supports one or two branches")
        if self.spline not in ("full", "fast", "discrete"):
            raise ValueError("Unknown spline implementation")
        if self.whole_field and self.wind_backend != "poam":
            raise ValueError("Whole-field experiments require the full POAM backend")
        if self.clarity_decay < 0 or self.clarity_decay_pattern not in (
            "uniform",
            "patchy",
            "striped",
        ):
            raise ValueError(
                "Nonnegative clarity process noise and a supported spatial pattern required"
            )
        if self.initial_clarity not in ("cold", "maintained"):
            raise ValueError("Unknown initial clarity condition")
        if (self.poam_support_x or self.poam_support_y) and min(
            self.poam_support_x, self.poam_support_y
        ) < 2:
            raise ValueError(
                "Both POAM support dimensions must be >=2, or both zero for automatic support"
            )
        if self.grid_nx < 2 or self.grid_ny < 2:
            raise ValueError("Monitoring grid dimensions must be >= 2")
        if self.wind_backend not in ("feature", "poam"):
            raise ValueError("Unknown wind backend")
        if self.control_representation not in ("acceleration", "waypoint"):
            raise ValueError("Unknown control representation")
        if self.branch_rule not in ("crosswind", "task"):
            raise ValueError("Unknown branch rule")
        if self.future_updates and self.future_steps % self.stage_steps:
            raise ValueError("Future updates require whole fixed-belief periods")


def load_config(path):
    data = yaml.safe_load(Path(path).read_text()) or {}
    profile = data.pop("profile", "smoke")
    for key in ("methods", "seeds"):
        if key in data:
            data[key] = tuple(data[key])
    presets = {
        "smoke": {},
        "dev": {
            "diffusion_steps": 10,
            "proposals": 32,
            "worlds": 6,
            "inner_samples": 10,
            "mission_horizons": 8,
        },
        "research": {
            "diffusion_steps": 24,
            "proposals": 96,
            "worlds": 12,
            "inner_samples": 24,
            "backup_steps": 24,
            "mission_horizons": 12,
            "seeds": tuple(range(20)),
        },
    }
    if profile not in presets:
        raise ValueError(f"Unknown profile {profile}")
    return replace(Config(), **(presets[profile] | data))


def config_dict(cfg):
    return asdict(cfg)
