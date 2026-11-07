"""Keep the learning phase outside a requested post-handoff MBD duration."""
from dataclasses import dataclass
from math import ceil


def block_capacity(seconds,commit_seconds):
    """Legacy allocation horizon; execution still ends at the exact schedule.

    A survey can finish between five-second MBD boundaries. Its matched total
    therefore need not be divisible by five. Pad only the legacy configuration
    capacity, never the actual experiment duration or the physics timestep.
    """
    if seconds<=0 or commit_seconds<=0:raise ValueError('Positive capacity and block length required')
    return ceil(seconds/commit_seconds)*commit_seconds

@dataclass
class MissionSchedule:
    total_seconds: float
    survey_limit: float
    monitoring_seconds: float | None = None
    gate_time: float | None = None

    def __post_init__(self):
        if self.total_seconds <= 0 or self.survey_limit <= 0:
            raise ValueError('Mission and survey limits must be positive')
        if self.monitoring_seconds is not None and self.monitoring_seconds <= 0:
            raise ValueError('Post-handoff monitoring duration must be positive')

    @property
    def end(self):
        if self.monitoring_seconds is None:return self.total_seconds
        return (self.survey_limit if self.gate_time is None else self.gate_time)+self.monitoring_seconds

    def handoff(self,time):
        if self.gate_time is not None:raise ValueError('Handoff must occur only once')
        if not 0 <= time <= self.survey_limit:raise ValueError('Handoff outside survey limit')
        self.gate_time=float(time)

    def completed(self,time):
        if self.monitoring_seconds is not None and self.gate_time is None:return False
        return abs(time-self.end)<1e-7
