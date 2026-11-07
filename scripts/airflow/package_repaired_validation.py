"""Audit and append new recordings without replacing previous demo content."""
from pathlib import Path
import gzip,hashlib,json,re,shutil,xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/gp_safety_20261104'
SITE=ROOT.parent/'sites/belief-adaptive-monitoring/dist';DEST=SITE/'revised/data/gp-safety-20261104'

def wind(scenario):
    folder=SITE/f'revised/data/scenario{scenario}';meta=json.loads((folder/'metadata.json').read_text());nx,ny,_=meta['grid']
    raw=np.fromfile(folder/meta['data_file'],dtype='<f4').reshape(ny,nx,3).astype(float);dx=meta['dx_m']
    def evaluate(p):
        z=np.clip(np.asarray(p)/dx-.5,0,[nx-1,ny-1]);i,j=np.minimum(np.floor(z).astype(int),[nx-2,ny-2]);fx,fy=z-[i,j]
        return (1-fy)*((1-fx)*raw[j,i,:2]+fx*raw[j,i+1,:2])+fy*((1-fx)*raw[j+1,i,:2]+fx*raw[j+1,i+1,:2])
    return evaluate

def main():
    environment=json.loads((OUT/'cold300/environment.json').read_text());free=np.array(environment['free_indices'])
    nx,ny=environment['grid'];dx=environment['dx_m'];xx,yy=np.meshgrid((np.arange(nx)+.5)*dx,(np.arange(ny)+.5)*dx);cells=np.c_[xx.ravel(),yy.ravel()]
    truth={s:np.array([wind(s)(p) for p in cells]) for s in (1,2)}
    audit=[];groups=[]
    for kind,folders,title,description in [
        ('cold',['cold300','cold300-s2'],'1 · No survey: uncertainty waypoints + RPCBF','Starts with no wind measurements. The robot selects uncertainty waypoints and receives actual local measurements. MBD requires whole-map RMSE ≤ 0.20 m/s. These are bounded tests of requested 300-second missions; the exact completed durations and stopping reasons are shown.'),
        ('informed',['informed300','informed300'],'2 · Accurate learned map: MBD + RPCBF','Starts from the GP fitted to 2,361 noisy full-coverage measurements in the unrestricted diagnostic. This is not a flown survey and is not known wind: MBD and RPCBF receive the learned posterior. Live measurements continue. The one-second warm planning limit remains enforced.')]:
        runs=[];outcomes=[]
        for s,folder in enumerate(folders,1):
            source=OUT/folder/f's{s}-center-baseline1.json.gz';r=json.load(gzip.open(source));sm=r['summary']
            assert sm['completed'] or sm['stop_reason'], 'Do not publish a still-running checkpoint as final'
            states=np.asarray(r['samples']);obs=np.asarray(r['observations']);mu=np.asarray(r['belief_frames'][-1]['mean']);actual=truth[s]
            assert np.isfinite(states).all() and np.max(np.abs(states[:,5:7]))<=1.6+1e-6
            np.testing.assert_allclose(obs[:,1:3],states[np.rint(obs[:,0]/.05).astype(int),1:3],atol=1e-6,rtol=0)
            err=np.linalg.norm(mu-actual,axis=1);recomputed=float(np.sqrt(np.mean(err[free]**2)))
            assert abs(recomputed-sm['final_rmse_m_s'])<1e-5
            # Independent NumPy integration of the unchanged quadratic drag law.
            field=wind(s)
            def f(x,u):
                relative=field(x[:2])-x[2:]
                return np.r_[x[2:],u+(.1/.680)*np.linalg.norm(relative)*relative]
            def step(x,u):
                h=.01;k1=f(x,u);k2=f(x+h*k1/2,u);k3=f(x+h*k2/2,u);k4=f(x+h*k3,u)
                return x+h*(k1+2*k2+2*k3+k4)/6
            errors=[]
            eligible=np.flatnonzero((states[:-1,18]>0)&(states[1:,18]>0))
            for k in eligible[::max(1,len(eligible)//100)]:
                expected=states[k,1:5].copy()
                for _ in range(5):expected=step(expected,states[k+1,5:7])
                errors.append(float(np.max(np.abs(expected-states[k+1,1:5]))))
            assert max(errors,default=0)<1e-6
            distance=np.linalg.norm(cells[:,None]-obs[None,:,1:3],axis=2).min(axis=1)
            near=free[distance[free]<=.2];far=free[distance[free]>.2]
            audit.append(dict(group=kind,scenario=s,finite=True,commands_within_authority=True,measurements_at_robot_positions=True,
                rk4_checked_segments=len(errors),rk4_max_state_difference=max(errors,default=0),rmse_recalculation_error=abs(recomputed-sm['final_rmse_m_s']),
                free_cells_within_20cm_of_executed_measurement=len(near),total_free_cells=len(free),
                near_measurements_rmse_m_s=float(np.sqrt(np.mean(err[near]**2))) if len(near) else None,
                beyond_measurements_rmse_m_s=float(np.sqrt(np.mean(err[far]**2))) if len(far) else None,
                final_safety_disk_coverage=r['belief_frames'][-1]['safety_disk_coverage'],summary=sm))
            # The full checkpoint stays in output. The browser needs trajectories,
            # maps, decisions and measurements, not 768x768 inference matrices.
            if r['handoff']:
                r['handoff']={k:v for k,v in r['handoff'].items() if k not in ('posterior_covariance','posterior_mean','inducing','clarity')}
            filename=f'{kind}-s{s}.json.gz'
            with gzip.GzipFile(DEST/filename,'wb',mtime=0) as file:file.write(json.dumps(r,separators=(',',':'),allow_nan=False).encode())
            shutil.copyfile(source.with_name(source.name.replace('.json.gz','-protocol.json')),DEST/f'{kind}-s{s}-protocol.json')
            runs.append(filename);outcomes.append(sm)
        if kind=='cold':result='Neither unrestricted GP accuracy nor low reported uncertainty certifies safe coverage. Check the recorded infeasible commands, coverage and error; a partial collision-free run is not a completed safe mission.'
        else:result='These bounded MBD attempts stopped at the planning-time limit. Their timings overlapped other test processes; use the isolated benchmark below for the runtime conclusion. No successful 300-second learned-wind MBD mission is claimed.'
        groups.append(dict(id='gpv-'+kind,title=title,description=description,runs=runs,result=result))
    cases=[]
    for path in sorted((OUT/'tests').glob('*.xml')):
        for c in ET.parse(path).iter('testcase'):
            failed=c.find('failure');error=c.find('error')
            cases.append(dict(name=c.attrib['classname']+'::'+c.attrib['name'],status='failed' if failed is not None or error is not None else 'passed',
                message=(failed if failed is not None else error).attrib.get('message') if failed is not None or error is not None else None))
    for path in sorted((OUT/'continuation-retry').glob('*.xml')):
        for c in ET.parse(path).iter('testcase'):cases.append(dict(name=c.attrib['classname']+'::'+c.attrib['name'],status='failed' if c.find('failure') is not None else 'passed'))
    retry=json.loads((OUT/'continuation-retry/results.json').read_text());timeouts=[r['file'] for r in retry if r['exit_code']=='timeout']
    tests=dict(collected=448,passed=sum(c['status']=='passed' for c in cases),failed=sum(c['status']=='failed' for c in cases),timed_out=len(timeouts),timeouts=timeouts,cases=cases)
    benchmark=json.loads((OUT/'isolated-benchmark/report.json').read_text())
    warm=[r for s in benchmark for r in s['rows'] if not r['compilation_included']]
    lo=min(r['total_s'] for r in warm);hi=max(r['total_s'] for r in warm)
    overview=f'<p class="gpv-note gpv-warning"><strong>The full system is not validated yet.</strong> The GP passes the unrestricted mean-accuracy test, but robot-path coverage, confidence calibration, filter feasibility and the one-second planning budget remain separate requirements.</p><p>Full suite: <strong>{tests["passed"]} passed, {tests["failed"]} failed, {tests["timed_out"]} timed out</strong>. New results are appended here; earlier recordings are preserved.</p>'
    timingrows=''.join(f'<tr><td>{s["scenario"]}</td><td>{r["repeat"]}</td><td>{r["conditioning_s"]:.3f}</td><td>{r["kernel_training_s"]:.3f}</td><td>{r["field_construction_s"]:.3f}</td><td>{r["geometry_s"]:.3f}</td><td>{r["mbd_search_and_validation_s"]:.3f}</td><td><b>{r["total_s"]:.3f}</b></td></tr>' for s in benchmark for r in s['rows'] if not r['compilation_included'])
    startups=json.loads((OUT/'startup-checks.json').read_text());shutil.copyfile(OUT/'startup-checks.json',DEST/'startup-checks.json')
    startup_ok=sum(r['feasible'] for r in startups['checks'])
    evidence=f'''<p><b>Isolated 15-second trajectory benchmark: {lo:.2f}–{hi:.2f} seconds per call on CPU.</b> Compilation is excluded. Kernel-training and observation-only calls are shown separately. No GPU runtime is claimed.</p><div class="gpv-scroll"><table><thead><tr><th>Scenario</th><th>Repeat</th><th>Condition</th><th>Train</th><th>Fields</th><th>Geometry</th><th>MBD + validation</th><th>Total (s)</th></tr></thead><tbody>{timingrows}</tbody></table></div>
<p><b>Startup filter check: {startup_ok}/{len(startups['checks'])} feasible first commands</b> across all six declared initial conditions, both scenarios and both initial beliefs. Every command is finite and within actuator limits. This checks startup only; later infeasible commands in the recordings show why it cannot certify a mission. <a href="data/gp-safety-20261104/startup-checks.json">Every startup state and command</a>.</p>
<p>The refined sampler uses the same POAM posterior and 32-node spectral quadrature, with 2,607 spatial support points. Measured covariance error falls from 49–52% to 1.3–1.4%. The cubic cache is refined from 0.2 to 0.1 m, reducing tested interpolation RMSE to 0.011 m/s. These are numerical refinements; the double-integrator state, quadratic aerodynamic force, fixed wind field, clarity equations, waypoint policy and MBD chains are unchanged.</p>
<p>RPCBF retains two coherent backup fields, a 1.2-second backup horizon, the full spline minimum, componentwise ±1.6 m/s² control limits, the existing pointwise 3√(σ²+0.1²) wind disk, and robust wall constraints. The bound is statistical and is not calibrated everywhere. An infeasible hard QP uses the existing logged minimum-slack fallback. These tests do not establish a global safety guarantee.</p>
<p>The five unit-test failures are four archived-demo assertions and a configuration test that tries to load environment YAML as planner configuration. The continuation cases using 0.1-second observations exceeded their individual 120-second limits. Failures are retained rather than weakening their assertions.</p>
<p><a href="data/gp-safety-20261104/test-report.json">Every test and failure</a> · <a href="data/gp-safety-20261104/audit.json">Independent recording audit</a> · <a href="data/gp-safety-20261104/benchmark.json">Exact runtime breakdown</a> · <a href="data/gp-safety-20261104/sampling.json">Sampler refinement</a></p>'''
    for name,data in [('environment.json',environment),('audit.json',audit),('test-report.json',tests),('benchmark.json',benchmark),('manifest.json',dict(groups=groups,truth={k:v.tolist() for k,v in truth.items()},overview=overview,evidence=evidence))]:
        (DEST/name).write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
        (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False))
    shutil.copyfile(OUT/'sampling/dense-support-refinement.json',DEST/'sampling.json')
    index=SITE/'revised/index.html';html=index.read_text()
    script='<!-- gp-safety-script:start --><script src="gp-safety.js" defer></script><!-- gp-safety-script:end -->'
    if script not in html:html=html.replace('</body>',script+'</body>',1);index.write_text(html)
    original=json.loads((OUT/'preservation-before.json').read_text());changed=[name for name,h in original.items() if hashlib.sha256((SITE/name).read_bytes()).hexdigest()!=h]
    assert set(changed)=={'index.html','revised/index.html'},changed
    for name,snapshot,tags in [('index.html','root-index-before.html',['gp-safety-link']),('revised/index.html','revised-index-before.html',['gp-safety','gp-safety-style','gp-safety-script'])]:
        text=(SITE/name).read_text()
        for tag in tags:text=re.sub(r'<!-- '+tag+r':start -->.*?<!-- '+tag+r':end -->\n?','',text,flags=re.S)
        assert text==(OUT/snapshot).read_text(),name
    preservation=dict(unchanged_original_files=len(original)-2,additive_html_only=True,changed=changed)
    (OUT/'preservation-after.json').write_text(json.dumps(preservation,indent=2));(DEST/'preservation.json').write_text(json.dumps(preservation))
    print(json.dumps(dict(tests={k:v for k,v in tests.items() if k!='cases'},preservation=preservation,recordings=[a['summary'] for a in audit]),indent=2))

if __name__=='__main__':main()
