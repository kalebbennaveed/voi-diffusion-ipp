import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
from mission_schedule import MissionSchedule,block_capacity

def test_survey_does_not_consume_300_second_monitoring_stage():
    schedule=MissionSchedule(300,600,300)
    assert not schedule.completed(300)
    schedule.handoff(142)
    assert schedule.end==442
    assert not schedule.completed(300)
    assert not schedule.completed(441.8)
    assert schedule.completed(442)

def test_unreached_gate_never_claims_complete_mbd():
    schedule=MissionSchedule(300,600,300)
    assert not schedule.completed(600)
    assert not schedule.completed(900)

def test_matched_proposed_keeps_exact_baseline_total():
    baseline=MissionSchedule(300,600,300);baseline.handoff(178)
    proposed=MissionSchedule(baseline.end,600)
    proposed.handoff(82)
    assert proposed.end==478
    assert proposed.completed(478)
    assert proposed.end-proposed.gate_time==396

def test_duplicate_and_late_handoffs_rejected():
    schedule=MissionSchedule(300,600,300)
    with pytest.raises(ValueError):schedule.handoff(601)
    schedule.handoff(142)
    with pytest.raises(ValueError):schedule.handoff(150)

def test_non_block_aligned_mission_uses_padded_capacity_but_exact_execution_end():
    assert block_capacity(532,5)==535
    assert block_capacity(300,5)==300
    schedule=MissionSchedule(532,532)
    schedule.handoff(180)
    assert schedule.end==532 and schedule.completed(532)
    assert not schedule.completed(535)
