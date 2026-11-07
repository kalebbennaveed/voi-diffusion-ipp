"""Adopt a verified field pair without changing the robot or its references."""
import argparse,hashlib,json,re,shutil
from pathlib import Path
from package_review import package

ROOT=Path(__file__).resolve().parents[2]

def publish(stage,site):
    read=lambda p:json.loads(p.read_text())
    old=read(site/'data/robot/manifest.json');new=read(stage/'data/robot/manifest.json')
    validation=read(stage/'data/validation.json')
    assert len(new['cases'])==12 and len(new['timestep_refinement'])==24
    assert new['model']==old['model']
    core=[s for s in old['source_sha256'] if s.startswith('src/')]
    assert all(old['source_sha256'][s]==new['source_sha256'][s] for s in core)
    for c in new['cases']:
        assert read(site/f"data/robot/{c['id']}.json")['reference']==read(stage/f"data/robot/{c['id']}.json")['reference']
    central=[c for c in new['cases'] if c['route']=='central' and c['direction']=='B-A']
    for c in central:
        safe=c['rpcbf'];transit=safe['central_transit']
        assert c['unfiltered']['contact'] and safe['target_region_reached'] and not safe['contact']
        assert safe['min_clearance_m']>=.06 and safe['infeasible_steps']==0
        assert transit and transit['duration_s'] and transit['mean_forward_speed_m_s']<=.2
        assert transit['duration_s']>=4*c['still_air']['central_transit']['duration_s']
        assert all(e['route']=='central' for e in safe['passage_crossings'])
    upper=[c for c in new['cases'] if c['route']=='upper']
    for c in upper:
        for method in ('unfiltered','rpcbf'):
            r=c[method];assert r['target_region_reached'] and not r['contact']
            assert all(e['route']=='upper' for e in r['passage_crossings'])
    lower=[c for c in new['cases'] if c['route']=='lower' and c['scenario']==1]
    assert all(c['unfiltered']['target_region_reached'] and not c['unfiltered']['contact'] for c in lower)
    # Publish the measured design for review, not a blanket acceptance claim.
    # Quietness and tracking failures stay in the unchanged test report below.
    assert validation['scenarios'][1]['fixed_horizontal_mean_speeds_m_s']['lower_bypass']>3
    assert all(c['same_outcome'] for c in new['timestep_refinement'])
    for s in (1,2):
        p=stage/f'data/scenario{s}/stationary-velocity.f32'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==new['wind_sha256'][str(s)]
    test_text=(stage/'data/directional-tests.txt').read_text()
    number=lambda name:int(re.search(r'(\d+) '+name,test_text).group(1)) if re.search(r'(\d+) '+name,test_text) else 0
    assert 'short test summary info' in test_text or ' passed in ' in test_text
    stats=dict(passed=number('passed'),failed=number('failed'),command='ROBOT_REVIEW_SITE='+str(stage)+' PYTHONPATH=src .venv/bin/python -m pytest tests/test_airflow_robot_review.py tests/test_rpcbf.py -q')
    min_lower=min(c['unfiltered']['min_clearance_m'] for c in lower)
    max_upper=max(c[m]['max_tracking_error_m'] for c in upper for m in ('unfiltered','rpcbf'))
    max_goal=max(c[m]['endpoint_error_m'] for c in upper for m in ('unfiltered','rpcbf'))
    qa=dict(all_design_targets_met=stats['failed']==0,robot_model_unchanged=True,all_12_reference_records_unchanged=True,
        core_physics_and_rpcbf_source_hashes_unchanged=True,all_24_timestep_refinements_same_outcome=True,
        max_timestep_position_difference_m=max(c['max_common_time_position_difference_m'] for c in new['timestep_refinement']),
        tests=stats,required_central_return_checks_passed=True,
        central_transits={str(c['scenario']):c['rpcbf']['central_transit'] for c in central},
        upper_all_eight_runs_reach_without_contact=True,scenario_one_lower_unfiltered_min_clearance_m=min_lower,
        strict_design_measurements=dict(monitoring_regions=[s['monitoring_regions'] for s in validation['scenarios']],
            upper_max_tracking_error_m=max_upper,upper_max_goal_error_m=max_goal,
            scenario_one_lower_max_tracking_error_m=max(c['unfiltered']['max_tracking_error_m'] for c in lower)),
        qualification='A long throat transit may include near-stops or backward motion. The speed plots and gate speeds retain this variability. Strict quietness and tracking checks are reported separately from the required central-slowdown and contact checks.')
    (stage/'data/directional-qa.json').write_text(json.dumps(qa,indent=2))
    lines=['STATIONARY CENTRAL-HEADWIND REVIEW','',
        'Selected by fixed-reference robot response, before interpreting monitoring-method performance.',
        'No interior field masks, speed caps, force multipliers, robot changes or reference retiming.',
        'The nominal planar double integrator and its existing quadratic relative-air force are unchanged.',
        'Both central B->A unfiltered runs contact; RPCBF traverses the actual central passage.',
        'Throat timing measures 1.2 m of net progress, including pauses/backtracking. It is not constant speed.',
        '',json.dumps(qa,indent=2),'','Reproduce the physical fields with Python 3.11 and requirements.txt:']
    prefixes=[]
    for s in (1,2):
        meta=read(stage/f'data/scenario{s}/metadata.json');o=meta['openings'];c=o['central_inlet'];floor=o['outlet']
        output=meta['generator']['source_directory'];prefixes.append(Path(output).name.split(f'_s{s}_n')[0])
        command=f"python scripts/airflow/simulate_sheltered_airflow.py --nx 64 --scenario {s} --seconds 32 --sample .1 --central-peak {c['peak_m_s']} --central-x {sum(c['x_m'])/2} --lower-peak 18 --floor-outlet --floor-box {' '.join(map(str,floor['x_m']+floor['y_m']))}"
        if 'lower_outlet' in o:command+=' --south-outlet'
        lines.extend([meta['wind_source'],meta['boundary_conditions'],command+' --output '+output,''])
    lines.extend(['python scripts/airflow/export_sheltered_wind.py --source-prefixes '+' '.join(prefixes)+' --site scenario-site',
        'PYTHONPATH=src python scripts/airflow/robot_review.py --site scenario-site','',
        'CFD: 64x37x16, dx=.125m, D3Q19 MRT, Smagorinsky Cs=.16, actual air molecular viscosity.',
        'Fixed vector mean over16..32s, exported at z=.75m. No time-varying wind is played.',
        'Full-scene grid convergence and experimental accuracy remain unestablished. This is a designed, provisional CFD environment.'])
    (ROOT/'scripts/airflow/DIRECTIONAL_REVIEW.txt').write_text('\n'.join(lines)+'\n')
    package(stage)
    for name in ('scenario1','scenario2','robot'):shutil.copytree(stage/'data'/name,site/'data'/name,dirs_exist_ok=True)
    for name in ('validation.json','directional-qa.json','directional-tests.txt','airflow-source.zip'):shutil.copyfile(stage/'data'/name,site/'data'/name)
    shutil.copyfile(stage/'app.js',site/'app.js')
    text=(site/'index.html').read_text()
    detail=f'''<details class="design-deviations"><summary>Measured quietness and tracking checks</summary><p>All eight upper runs reach their targets without contact. Maximum upper tracking error is {max_upper:.3f} m and maximum final goal error is {max_goal:.3f} m. The coupled CFD field changes when its boundary conditions change.</p><p>Scenario 1’s lower passage has {validation['scenarios'][0]['fixed_horizontal_mean_speeds_m_s']['lower_bypass']:.3f} m/s mean horizontal wind. Both unfiltered lower runs reach their target; minimum clearance is {min_lower:.3f} m. The complete route still encounters approach flow. Regional wind measurements and stricter quietness/tracking limits are retained in the reports.</p><p><a href="data/directional-qa.json">Design-check report ↗</a> · <a href="data/directional-tests.txt">{stats['passed']} tests passed; {stats['failed']} stricter design checks failed ↗</a></p></details>'''
    text=re.sub(r'<details class="design-deviations">.*?</details>',detail,text,flags=re.S)
    if 'id="throat-definition"' not in text:
        text=text.replace('<details class="design-deviations">','<p id="throat-definition" class="robot-caption">Central throat timing uses fixed planes 0.6 m either side of the passage gate. It counts pauses and backward motion during the 1.2 m crossing. Speed varies along the path; the forward-speed plots and gate-speed readouts show that variation.</p><details class="design-deviations">')
    (site/'index.html').write_text(text)
    print(json.dumps(qa,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--site',type=Path,required=True)
    a=p.parse_args();publish(a.stage,a.site)
