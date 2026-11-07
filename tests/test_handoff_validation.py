import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
from handoff_validation import forecast_passes


def test_task_accuracy_is_explicit_and_does_not_hide_contact_or_bad_cost():
    assert not forecast_passes(True,True,0,.35,1.1,.004)
    assert forecast_passes(True,True,8,.35,1.1,.004,True,'monitoring-cost')
    assert not forecast_passes(True,True,8,.35,1.1,.004,False,'monitoring-cost')
    assert not forecast_passes(True,False,0,.01,.02,.001,True,'monitoring-cost')
    assert not forecast_passes(True,True,0,.01,.02,.11,True,'monitoring-cost')
    assert not forecast_passes(False,True,0,.01,.02,.001,True,'monitoring-cost')
    assert not forecast_passes(True,True,0,float('nan'),.02,.001,True,'monitoring-cost')
