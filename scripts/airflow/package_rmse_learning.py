"""Append an RMSE-gated diagnostic, preserving all existing simulation files."""
import json,gzip,hashlib,shutil,re
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import rmse_monitoring as im
import cluster_environment as ce

root=im.m.physical.ROOT
source=root/'results/airflow/rmse_learning_20261104/checked300'
site=im.m.physical.SITE/'revised';dest=site/'data/rmse-learning-300'
output=root/'output/rmse_gate_20261104'
if dest.exists():raise RuntimeError('Do not overwrite published experiment data')
existing={str(f.relative_to(site)):hashlib.sha256(f.read_bytes()).hexdigest() for f in site.rglob('*') if f.is_file() and f.name!='rmse-learning.js'}
(output/'preservation-before.json').write_text(json.dumps(existing,indent=2))
dest.mkdir(parents=True)
records=[]
for f in sorted(source.glob('s*-baseline1.json.gz')):
 r=json.load(gzip.open(f));records.append(r)
for f in source.glob('*.json*'):shutil.copyfile(f,dest/f.name)
env,grid,_=ce.environment(im.legacy.physical_settings(im.settings()),root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
truth={s:np.asarray(im.m.physical.load_wind(s,site).at(jnp.asarray(grid['all_cells']))).round(6).tolist() for s in (1,2)}
runs=[];audit=[]
for r in records:
 samples=np.asarray(r['samples'],dtype=float);obs=np.asarray(r['observations'])
 np.testing.assert_allclose(obs[:,1:3],samples[np.rint(obs[:,0]/.05).astype(int),1:3],atol=1e-6,rtol=0)
 assert np.max(np.abs(samples[:,5:7]))<=1.6+1e-6
 assert np.isfinite(samples).all()
 threshold=r['summary']['rmse_threshold_m_s']
 eligible=[b for b in r['belief_frames'] if b['rmse_m_s']<=threshold]
 assert r['summary']['gate_reached']==bool(eligible)
 assert bool([p for p in r['plans'] if p['phase']=='monitoring'])==r['summary']['gate_reached']
 errors=[]
 for b in r['belief_frames']:
  idx=grid['free_indices'];error=np.asarray(b['mean'])[idx]-np.asarray(truth[r['scenario']])[idx]
  errors.append(abs(np.sqrt(np.mean(np.sum(error**2,axis=1)))-b['rmse_m_s']))
 assert max(errors)<1e-5
 audit.append(dict(scenario=r['scenario'],finite_states_commands_slack=True,real_measurement_positions=True,
    maximum_rmse_recalculation_error_m_s=max(errors),gate_matches_measured_error=True,
    safety_success=r['summary']['contact'] is None,learning_accuracy_success=bool(eligible)))
 runs.append(dict(scenario=r['scenario'],file=r['id']+'.json.gz',summary=r['summary']))
(dest/'manifest.json').write_text(json.dumps(dict(runs=runs,truth=truth,threshold_m_s=.2),separators=(',',':')))
(dest/'audit.json').write_text(json.dumps(audit,indent=2))
index=site/'index.html';original=index.read_text()
first='''<!-- rmse-learning:start -->
<section id="wind-learning-rmse" class="learning-review" aria-labelledby="rmse-learning-title">
<p class="eyebrow">Accuracy check · Stage 1 · actual measurements only</p><h2 id="rmse-learning-title">Learn wind with uncertainty waypoints and RPCBF</h2>
<p class="learning-intro">The robot selects high-uncertainty waypoints and executes through RPCBF. The orange curve shows wind-vector RMSE against the true stationary wind over every free-space grid cell. MBD starts only when this error reaches <b>0.20 m/s or less</b>. This threshold is an accuracy condition, not a safety guarantee.</p>
<div class="rmse-view"><p>Loading measured wind error…</p></div>
<p class="learning-note learning-warning"><b>The learning criterion is not met in either scenario.</b> Scenario 1 stops at boundary contact at 103.31 s; Scenario 2 completes 300 s without contact. Reported uncertainty falls while whole-map error remains high. The MBD stage below therefore has no executed trajectory.</p>
<details><summary>What changed, and which approximations remain</summary>
<p>The robot, aerodynamic force, stationary physical wind and clarity equations are unchanged. The experiment fixes a floating-point edge bug in the backup wind cache and adds four robust wall-braking constraints alongside RPCBF. Infeasible commands still use the logged minimum-slack fallback; this is not a guarantee of safety.</p>
<p>POAM remains a sparse variational GP with 128 inducing variables, an attentive kernel and its full predictive residual. Field draws use the existing 32-node spectral quadrature with support correction. The controller uses the existing 0.2 m cubic wind cache, 1.2 s backup horizon and a pointwise 3σ bound. These are numerical/model approximations, and the current uncertainty is demonstrably miscalibrated away from measured paths.</p>
<p>Only the simulator and this diagnostic use the true field. Waypoint choice, POAM inference, MBD and RPCBF receive actual local measurements and the learned belief. The phase switch receives only the RMSE decision.</p>
<p><a href="data/rmse-learning-300/audit.json">Recorded-data audit</a> · <a href="data/rmse-learning-300/summary.json">Full outcomes</a> · <a href="#mbd-rmse">Separate MBD stage</a></p></details>
</section><!-- rmse-learning:end -->
'''
second='''<!-- rmse-mbd:start -->
<section id="mbd-rmse" class="learning-review" aria-labelledby="rmse-mbd-title">
<p class="eyebrow">Baseline 1 · Stage 2 · begins after sufficient wind accuracy</p><h2 id="rmse-mbd-title">MBD after wind learning</h2>
<p class="learning-intro">This section contains only the MBD phase. Learning is shown separately above. The handoff preserves robot position, velocity, clarity and the learned posterior. It does not substitute the true wind for the learned wind.</p>
<div class="rmse-view"><p>Loading the MBD handoff status…</p></div>
<details><summary>MBD configuration and why this phase has not started</summary>
<p>The optimizer retains the original independent geometric MBD chains, all 120 acceleration coordinates and the 15 s horizon. It evaluates full, unattenuated coherent POAM posterior draws: the earlier unsupported-region suppression is removed from this baseline. The monitoring objective is unchanged. There are 43 original search evaluations plus seven mode-leader rescores; the global winner and best distinct optimized mode leader receive independent hard-filter validation. An unvalidated fallback is not executed as an MBD plan.</p>
<p>The relaxed filter is used during search with 0.25 s forecasting; finalist checks use the hard filter and the existing 0.05 s control / 0.01 s integration. This is the declared numerical resolution difference. Actual wind measurements continue between planning calls.</p>
<p>No MBD mission is reported here because neither learning run reached 0.20 m/s RMSE. The earlier fixed-time learning/MBD recordings remain below, unchanged.</p></details>
</section><!-- rmse-mbd:end -->
'''
assert '<!-- rmse-learning:start -->' not in original
changed=original.replace('<!-- waypoint-review:start -->',first+'<!-- waypoint-review:start -->',1)
changed=changed.replace('<!-- baseline1-review:start -->',second+'<!-- baseline1-review:start -->',1)
changed=changed.replace('</nav>','<!-- rmse-nav:start --><a href="#wind-learning-rmse">Wind RMSE · learning</a><a href="#mbd-rmse">MBD after learning</a><!-- rmse-nav:end --></nav>',1)
changed=changed.replace('</body>','<!-- rmse-script:start --><script src="rmse-learning.js" defer></script><!-- rmse-script:end --></body>',1)
restored=changed
for tag in ['rmse-learning','rmse-mbd','rmse-nav','rmse-script']:
 restored=re.sub(r'<!-- '+tag+r':start -->.*?<!-- '+tag+r':end -->\n?','',restored,flags=re.S)
assert restored==original
index.write_text(changed)
modified=[p for p,h in existing.items() if hashlib.sha256((site/p).read_bytes()).hexdigest()!=h]
assert modified==['index.html'],modified
(output/'preservation-after.json').write_text(json.dumps(dict(unchanged_files=len(existing)-1,changed=modified,only_additive_html=True),indent=2))
print(json.dumps(dict(audit=audit,preserved=len(existing)-1)))
