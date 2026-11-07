"""Independently audit survey+300s baseline versus equal-duration masked MBD."""
from pathlib import Path
import json,gzip,hashlib
import numpy as np
from audit_full_learning import audit
from evaluate_task_learning import evaluate,stats

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/matched_duration_20261105'

def main():
    cohorts={};recordings={};inventory=[];protocols=[]
    for kind,label in [('baseline','Baseline 1'),('proposed','Clarity-masked learning')]:
        folder=OUT/kind
        paths=sorted(folder.glob('s*-center-*.json.gz'))
        if len(paths)!=2:raise RuntimeError(f'{kind}: need both scenarios')
        checks=audit(folder);rows=evaluate(folder)
        for row,path in zip(rows,paths):
            r=json.load(gzip.open(path));sm=r['summary'];recordings[kind,row['scenario']]=r
            assert sm['completed'] and sm['gate_reached']
            assert r['handoff_validation'] and r['handoff_validation'][-1]['passed']
            gate=sm['gate_time_s'];handoff=r['handoff'];s=np.array(r['samples'])
            assert handoff['t']==gate
            np.testing.assert_allclose(handoff['state'],s[round(gate/.05),1:5],atol=1e-7)
            q=next(f['q'] for f in r['clarity_frames'] if f['t']==gate)
            np.testing.assert_allclose(handoff['clarity'],q,atol=5.01e-7)
            assert handoff['observations']==1+round(gate/.2)
            gate_check=r['handoff_validation'][-1]
            assert gate_check['t']==gate
            for case in gate_check['cases']:
                assert case['passed'] and not case['contact'] and case['infeasible_steps']==0
                assert case['position_rmse_m']<=.25 and case['max_position_error_m']<=.5
                assert case['relative_monitoring_cost_error']<=.10
            assert any(c['displacement_m']>=.5 for c in gate_check['cases'])
            assert not any(p.get('execution_mode')=='targeted wind learning' for p in r['plans'])
            if kind=='baseline':assert sm['monitoring_duration_s']==300 and sm['simulated_seconds']==gate+300
            row.pop('complete_300',None)
            row.update(completed=sm['completed'],survey_seconds=sm['survey_duration_s'],
                phase_monitoring_seconds=sm['monitoring_duration_s'],gate_validation=gate_check,
                handoff_state_clarity_and_measurements_preserved=True,whole_mission_wall_seconds=sm['wall_seconds'],
                warm_mbd_requests_over_two_seconds=sum(sum(p['timing'][k] for k in ('fields','geometry','optimization'))>2 for p in r['plans'] if p['phase']=='monitoring' and not p['compilation_included']),
                warm_mbd_full_cycles_over_two_seconds=sum(p['wall_seconds']>2 for p in r['plans'] if p['phase']=='monitoring' and not p['compilation_included']),
                handoff_diagnostic_seconds=sum(c['wall_seconds'] for c in r['handoff_validation']),
                startup_mbd_seconds=sum(p['wall_seconds'] for p in r['plans'] if p['phase']=='monitoring' and p['compilation_included']))
            row.pop('warm_actual_decision_deadline_misses',None)
            protocols.append(json.loads(path.with_name(path.name.replace('.json.gz','-protocol.json')).read_text()))
        (folder/'numerical-audit.json').write_text(json.dumps(checks,indent=2,allow_nan=False))
        (folder/'comparison.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        cohorts[kind]=dict(label=label,runs=rows)
        for path in folder.iterdir():
            if path.suffix in ('.json','.gz','.npz','.zip','.pkl'):
                with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                inventory.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest))
    # Only elapsed mission length differs in physical configuration.
    common=('settings','gp_profile','sensor_period_s','control_period_s','integration_period_s','noise_std_m_s','safety')
    for key in common:assert all(p[key]==protocols[0][key] for p in protocols),key
    physical=lambda p:{k:v for k,v in p['physical_settings'].items() if k!='mission_seconds'}
    assert all(physical(p)==physical(protocols[0]) for p in protocols)
    names={'full_learning_monitoring.py','rmse_monitoring.py','efficient_learning_monitoring.py','spatial_poam.py','waypoint_poam.py','exact_poam_variance.py','cached_poam_fields.py','supported_poam.py','reachable_learning.py','uncertainty_waypoints.py','guarded_waypoints.py','monitoring_routes.py','task_wind_acquisition.py','handoff_validation.py','mission_schedule.py'}
    shared=[name for name in protocols[0]['source_sha256'] if name.startswith('src/') or Path(name).name in names]
    for name in shared:assert all(p['source_sha256'][name]==protocols[0]['source_sha256'][name] for p in protocols),name
    contrasts=[]
    for scenario in (1,2):
        base=next(r for r in cohorts['baseline']['runs'] if r['scenario']==scenario)
        prop=next(r for r in cohorts['proposed']['runs'] if r['scenario']==scenario)
        assert base['simulated_seconds']==prop['simulated_seconds']
        # Identical sensor-noise seeds, with actual sensor positions allowed to differ.
        b=recordings['baseline',scenario];p=recordings['proposed',scenario]
        assert len(b['observations'])==len(p['observations'])
        contrasts.append(dict(scenario=scenario,matched_total_seconds=base['simulated_seconds'],
            baseline_survey_seconds=base['survey_seconds'],proposed_survey_seconds=prop['survey_seconds'],
            baseline_monitoring_seconds=base['phase_monitoring_seconds'],proposed_monitoring_seconds=prop['phase_monitoring_seconds'],
            baseline_cost=base['full_mission_monitoring_cost'],proposed_cost=prop['full_mission_monitoring_cost'],
            cost_reduction_percent=100*(base['full_mission_monitoring_cost']-prop['full_mission_monitoring_cost'])/base['full_mission_monitoring_cost'],
            no_contact_both=base['contact'] is None and prop['contact'] is None))
    payload=dict(cohorts=cohorts,contrasts=contrasts,validation=json.loads((OUT/'threshold-selection.json').read_text()),
        notes=['One initial condition and actual-measurement noise seed per scenario; no multi-seed robustness claim.',
            'No true wind in optimization, controller or inference. Simulator truth is authorized only for plant, actual local sensors, diagnostic errors and handoff validation.',
            'Baseline and proposed use different stopping domains. This comparison does not isolate acquisition weighting alone.',
            'No contact does not prove safety; infeasible commands, hold fallbacks and compute outliers are reported.',
            'The full global RMSE remains high where wind is difficult to learn; passing finite trajectory checks is not a globally accurate wind-map claim.'])
    (OUT/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    (OUT/'run-manifest.json').write_text(json.dumps(dict(inputs=inventory,shared_source_files=shared,models_identical=True,handoff_preservation_verified=True,duration_matching_verified=True),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for scenario in (1,2):
        for kind,color,label in [('baseline','#b65c19','Baseline 1'),('proposed','#175e91','Clarity mask')]:
            r=recordings[kind,scenario];samples=np.asarray(r['samples']);frames=r['belief_frames'];gate=r['summary']['gate_time_s']
            axes[0,scenario-1].plot([f['t'] for f in frames],[f['rmse_m_s'] for f in frames],color=color,label=label)
            axes[1,scenario-1].plot(samples[:,0],samples[:,15],color=color,label=label+' · A')
            axes[1,scenario-1].plot(samples[:,0],samples[:,16],color=color,ls='--',label=label+' · B')
            axes[2,scenario-1].plot(samples[:,0],samples[:,14],color=color,label=label)
            for ax in axes[:,scenario-1]:ax.axvline(gate,color=color,lw=.8,ls=':',alpha=.7)
        axes[0,scenario-1].set_title(f'Scenario {scenario} · {samples[-1,0]:g} s per method')
        for ax,ylabel in zip(axes[:,scenario-1],['Whole-map wind RMSE (m/s)','Mean region clarity','Accumulated monitoring cost']):
            ax.set_ylabel(ylabel);ax.set_xlabel('Elapsed simulation time (s)');ax.legend(fontsize=8);ax.grid(alpha=.15)
        axes[1,scenario-1].set_ylim(0,1)
    fig.suptitle('Survey + 300 s of MBD versus clarity-masked learning at equal total time\nDotted lines mark each method’s actual handoff',fontsize=13)
    fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    print(json.dumps(contrasts,indent=2))

if __name__=='__main__':main()
