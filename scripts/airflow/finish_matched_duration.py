"""Run the masked comparison only after both complete baseline durations exist."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/matched_duration_20261105'
start=time.monotonic()
while True:
    path=OUT/'baseline/summary.json'
    if path.exists():
        rows=json.loads(path.read_text())
        if any(not r['completed'] for r in rows):raise RuntimeError('Baseline failed; proposed batch not launched')
        if len(rows)==2:break
    if time.monotonic()-start>6600:raise TimeoutError('Baseline wait limit; proposed batch not launched')
    time.sleep(10)
print(json.dumps(dict(event='baseline_durations',durations={r['scenario']:r['simulated_seconds'] for r in rows})),flush=True)
command=[sys.executable,'scripts/airflow/run_full_learning.py','--output',str(OUT/'proposed'),
    '--match-duration-from',str(OUT/'baseline'),'--survey-seconds','900','--modes','routeprecision',
    '--validate-handoff','--gp-lengths','.1,.2,.5,1','--observation-first-inducing','--exact-safety-variance',
    '--safety-confidence','7','--require-completion','--train-kernel','--plan-limit','2','--wall-limit','3000','--efficient-planning',
    '--learning-controller','reachable-waypoints','--global-information-goals','--factorized-support',
    '--prefix-screen','braking-feasible','--diversify-stalled-routes','--save-profile']
with (OUT/'proposed-command.json').open('w') as f:json.dump(command,f,indent=2)
with open('/tmp/matched-proposed.log','w') as log:
    result=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
if result.returncode:raise RuntimeError(f'Proposed batch exited {result.returncode}; recordings retained, no completion claimed')
print(json.dumps(dict(event='proposed_complete')),flush=True)
