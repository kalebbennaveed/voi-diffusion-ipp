from dataclasses import dataclass, replace, asdict
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Config:
    scenario: str = 'corridor'
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
    wind_std: float = 0.55
    wind_length: float = 1.1
    observation_std: float = 0.10
    ig_weight: float = 0.025
    measurements: bool = True
    shared_future: bool = False
    independent_cells: bool = False
    spline: str = 'full'
    seeds: tuple = (0, 1, 2)
    methods: tuple = ('proposed', 'passive', 'weighted_ig', 'heuristic', 'cem', 'mppi', 'oracle')

    def __post_init__(self):
        for name in ('stage_steps', 'future_steps', 'knots', 'branches', 'worlds',
                     'inner_samples', 'backup_steps', 'diffusion_steps', 'proposals',
                     'candidate_chunk', 'mission_horizons'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.backup_steps < 3 or self.knots < 2:
            raise ValueError('Cubic spline requires >=3 backup steps; interpolation >=2 knots')
        if not 0 < self.beta_min <= self.beta_max < 1 or self.temperature <= 0:
            raise ValueError('Invalid diffusion schedule/temperature')
        if self.observation_std <= 0 or self.dt <= 0:
            raise ValueError('Positive observation noise and timestep required')
        if self.branches not in (1, 2):
            raise ValueError('Current observation router supports one or two branches')
        if self.spline not in ('full', 'fast', 'discrete'):
            raise ValueError('Unknown spline implementation')


def load_config(path):
    data = yaml.safe_load(Path(path).read_text()) or {}
    profile = data.pop('profile', 'smoke')
    for key in ('methods', 'seeds'):
        if key in data:
            data[key] = tuple(data[key])
    presets = {'smoke': {}, 'dev': dict(diffusion_steps=10, proposals=32, worlds=6,
               inner_samples=10, mission_horizons=8),
               'research': dict(diffusion_steps=24, proposals=96, worlds=12,
               inner_samples=24, backup_steps=24, mission_horizons=12, seeds=tuple(range(20)))}
    if profile not in presets:
        raise ValueError(f'Unknown profile {profile}')
    return replace(Config(), **(presets[profile] | data))


def config_dict(cfg):
    return asdict(cfg)
