"""Paired execution evidence for changing only the learning handoff domain."""
import gzip,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/'output/task_learning_20261105'


def main():
    reports=[]
    for scenario in (1,2):
        mask_path=ROOT/f'diversified/s{scenario}-center-mask.json.gz'
        route_path=ROOT/f'route_precision/s{scenario}-center-routeprecision.json.gz'
        if not mask_path.exists() or not route_path.exists():continue
        mask=json.load(gzip.open(mask_path));route=json.load(gzip.open(route_path));gate=route['summary']['gate_time_s']
        if gate is None:continue
        a=np.asarray(mask['samples']);b=np.asarray(route['samples']);n=round(gate/.05)+1
        np.testing.assert_array_equal(a[:n,:],b[:n,:])
        aobs=np.asarray(mask['observations']);bobs=np.asarray(route['observations']);nobs=round(gate/.2)+1
        np.testing.assert_array_equal(aobs[:nobs],bobs[:nobs])
        ap=json.loads(mask_path.with_name(mask_path.name.replace('.json.gz','-protocol.json')).read_text())
        bp=json.loads(route_path.with_name(route_path.name.replace('.json.gz','-protocol.json')).read_text())
        for key in ['settings','physical_settings','gp_profile','initial_condition']:
            assert ap[key]==bp[key],key
        assert (mask_path.parent/'environment.json').read_bytes()==(route_path.parent/'environment.json').read_bytes()
        complete=mask['summary']['completed'] and route['summary']['completed']
        report=dict(scenario=scenario,gate_s=gate,identical_robot_control_clarity_prefix=True,
            identical_actual_sensor_prefix=True,identical_model_configuration=True,
            source_scope='Clarity-mask survey versus route-precision handoff; not versus the global-survey baseline.',
            route_uncertainty_at_handoff=route['handoff']['route_precision']['vector_std_m_s'],
            complete_paired_300s=complete,mask_cost=mask['summary']['integrated_deficit'] if complete else None,
            route_cost=route['summary']['integrated_deficit'] if complete else None,
            mask_contact=mask['summary']['contact'],route_contact=route['summary']['contact'])
        if complete:report['cost_reduction_percent']=100*(1-report['route_cost']/report['mask_cost'])
        reports.append(report)
    (ROOT/'handoff-comparison.json').write_text(json.dumps(reports,indent=2,allow_nan=False))
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
