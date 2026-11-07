"""Add two recorded experiments without replacing any previous demo recording."""
import argparse,gzip,hashlib,json,shutil,re
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--site',type=Path,required=True);a=p.parse_args()
    dest=a.site/'data/waypoint-learning-300'
    if dest.exists():raise RuntimeError('Keep previous published data; use a fresh dataset name')
    runs=[];records={}
    for f in sorted(a.directory.glob('s*-*.json.gz')):
        with gzip.open(f,'rt') as h:r=json.load(h)
        records[(r['scenario'],r['method'])]=r
        runs.append(dict(id=r['id'],scenario=r['scenario'],method=r['method'],file=f.name,summary=r['summary']))
    assert len(runs)==4
    checks=[]
    for scenario in (1,2):
        survey=records[scenario,'survey'];baseline=records[scenario,'baseline1'];handoff=baseline['handoff']
        assert handoff is not None,'Baseline must actually enter its MBD phase'
        prefix=[r for r in survey['samples'] if r[0]<=handoff['t']]
        other=[r for r in baseline['samples'] if r[0]<=handoff['t']]
        assert prefix==other,'Matched learning prefixes must agree exactly'
        checks.append(dict(scenario=scenario,handoff_t=handoff['t'],identical_learning_prefix=True,
                           measurements_at_handoff=handoff['observations']))
    dest.mkdir(parents=True)
    for f in a.directory.glob('*.json*'):shutil.copyfile(f,dest/f.name)
    (dest/'manifest.json').write_text(json.dumps(dict(runs=runs,handoff_checks=checks),indent=2))
    index=a.site/'index.html';original=index.read_text();assert '<!-- waypoint-review:start -->' not in original
    first='''<!-- waypoint-review:start -->
<section id="uncertainty-waypoints" class="learning-review" aria-labelledby="waypoint-title">
<p class="eyebrow">Incremental experiment 1 · actual robot measurements · no MBD</p>
<h2 id="waypoint-title">Wind uncertainty waypoints with RPCBF</h2>
<p class="learning-intro">High-uncertainty waypoints drive the robot through the environment. The tracking controller sends every command through the existing full-uncertainty RPCBF. This is one integrated policy in each scenario.</p>
<div class="learning-view"><p role="status">Loading the executed wind-learning experiments…</p></div>
<p class="learning-note">Purple: current geometric path and selected waypoint. Blue map: full POAM uncertainty, on a fixed 0–1.8 m/s scale. Choose “Learned mean wind speed” to see the measured belief and its direction. Lower variance alone does not establish accuracy; offline error and infeasible filter commands are reported alongside it.</p>
<details><summary>Waypoint selection, actual sensing, and safety</summary>
<p>Every 0.2 s, actual noisy wind measurements update POAM. A geometric graph ranks reachable waypoints by posterior variance divided by (1 + path length / 1 m). Targets persist until arrival, sufficient local learning, or a measured lack of progress. Stalled targets receive a temporary 30 s cooldown. There is no clarity reward or MBD in this first experiment.</p>
<p>Edges whose predicted mean wind compensation exceeds 1.2 m/s² per component are excluded, preserving 0.4 m/s² of nominal feedback authority. This is a waypoint feasibility screen, not a safety guarantee. RPCBF still receives the full posterior uncertainty and operates every 0.05 s. Infeasible QPs and their minimum-slack fallback remain visible.</p>
<p>POAM retains its full predictive residual and 128 inducing variables, selected by its original pivoted-Cholesky rule from existing inducing locations and actual new measurements. Its existing variational EM objective trains kernel parameters with four Adam steps every 2 s, using only actual measurements. Prediction and sampling caches are refreshed after training. No map-wide synthetic survey or simulated true wind enters selection, inference, or control.</p>
<p><a href="data/waypoint-learning-300/manifest.json">Recorded outcomes and handoff checks</a> · <a href="data/waypoint-learning-300/summary.json">Full run summaries</a></p>
</details></section>
<!-- waypoint-review:end -->'''
    second='''<!-- baseline1-review:start -->
<section id="learn-then-mbd" class="learning-review" aria-labelledby="baseline1-title">
<p class="eyebrow">Incremental experiment 2 · Baseline 1</p>
<h2 id="baseline1-title">First learn wind, then disturbance-aware MBD</h2>
<p class="learning-intro">0–60 s: the same uncertainty-waypoint policy with RPCBF. 60–300 s: existing disturbance-aware MBD uses the learned POAM belief. The robot, velocity, clarity, measurements, and clock carry across the handoff. Learning time is included in the 300 s mission.</p>
<div class="learning-view"><p role="status">Loading Baseline 1…</p></div>
<p class="learning-note">The learned wind remains uncertain. MBD retains the 15 s horizon, independent geometric initializations, 43 search evaluations, two coherent performance fields and independent hard-filter finalist checks. Actual sensing continues during monitoring. The first 60 s are checked to match Experiment 1 exactly. This shorter handoff follows the recorded failure of the longer learning pilot; it is not an independently validated choice of learning budget.</p>
<details><summary>What is and is not established by these runs</summary><p>These are executed simulations from the center of Region B, using the same physical wind, robot, and clarity models as the comparisons above. The outcome table reports contact, visits to A, uncertainty reduction, offline wind error, and filter infeasibility. First-call compilation is excluded from warm decision timing and is recorded separately. Lower uncertainty and reaching A do not by themselves establish sustained monitoring success.</p></details>
</section>
<!-- baseline1-review:end -->'''
    marker='<section class="geometry-review"'
    assert marker in original
    changed=original.replace(marker,first+'\n'+marker,1)
    marker='<!-- retest-review:start -->'
    assert marker in changed
    changed=changed.replace(marker,second+'\n'+marker,1)
    changed=changed.replace('</head>','<!-- waypoint-assets:start --><link rel="stylesheet" href="waypoint-learning.css"><!-- waypoint-assets:end --></head>',1)
    changed=changed.replace('</body>','<!-- waypoint-scripts:start --><script src="waypoint-learning.js" defer></script><!-- waypoint-scripts:end --></body>',1)
    changed=changed.replace('</nav>','<!-- waypoint-nav:start --><a href="#uncertainty-waypoints">Uncertainty waypoints</a><a href="#learn-then-mbd">Learn then MBD</a><!-- waypoint-nav:end --></nav>',1)
    restored=changed
    for name in ('waypoint-review','baseline1-review','waypoint-assets','waypoint-scripts','waypoint-nav'):
        restored=re.sub(r'<!-- '+name+r':start -->.*?<!-- '+name+r':end -->\n?', '',restored,flags=re.S)
    assert restored==original,'The existing page must be preserved exactly after removing additive blocks'
    index.write_text(changed)
    print(json.dumps(dict(url='http://127.0.0.1:8786/revised/#uncertainty-waypoints',runs=runs,handoff_checks=checks)))


if __name__=='__main__':main()
