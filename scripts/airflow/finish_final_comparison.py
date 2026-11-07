"""Finish declared trials serially from verified checkpoints, then audit/export.

Wait for the existing batch before reconstructing missing inference checkpoints
so recovery does not compete with the main simulations. No new seeds, model
changes, outcome selection, or shortened mission horizons are introduced.
"""
import json,os,subprocess,sys,time
from pathlib import Path
from package_final_comparison import OUT,source,load

def execute(args,log,timeout):
    print(json.dumps(dict(event='start',log=str(log),args=args)),flush=True)
    with log.open('a') as f:
        result=subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,env=os.environ,timeout=timeout)
    print(json.dumps(dict(event='end',log=str(log),code=result.returncode)),flush=True)
    return result.returncode

def continuation_args(s,method,path,destination):
    r=load(path);p=load(path.with_name(r['id']+'-protocol.json'))
    args=['scripts/airflow/run_full_learning.py','--output',str(destination),'--scenarios',str(s),'--modes',r['method'],
        '--comparison-method',p['comparison_method'],'--seconds',str(int(p['total_seconds'])),
        '--survey-seconds',str(int(p['learning_limit_s'])),'--uncertainty-only','--efficient-planning',
        '--allow-validation-slack','--observation-first-inducing','--exact-safety-variance',
        '--gp-lengths','.1,.2,.5,1','--train-kernel','--factorized-support','--plan-limit','2',
        '--wall-limit','3600','--save-profile','--resume-monitoring-from',str(path),
        '--learning-controller',p['learning_controller'],'--prefix-screen',p['prefix_screen'],
        '--global-std-threshold','1.1','--task-std-threshold','1.1']
    if p['global_information_goals']:args.append('--global-information-goals')
    if p['diversify_stalled_routes']:args.append('--diversify-stalled-routes')
    if p['monitoring_seconds_after_handoff'] is not None:args+=['--monitoring-seconds',str(p['monitoring_seconds_after_handoff'])]
    if s==3:args+=['--environment-spec','configs/scenarios/calm_monitoring_20261106.json',
        '--wind-site',str(OUT/'scenario3-flow-v2'),'--initial-state','1.4','4.15','0','0','--initial-region-clarity','.35']
    return args

def main():
    began=time.monotonic()
    prerequisites=[OUT/x/'RUN_COMPLETE' for x in ('filtered-s3','global-s3','aware-s3')]
    while not all(p.exists() for p in prerequisites):
        if time.monotonic()-began>12000:raise TimeoutError('Queued batch did not finish; inspect per-job logs')
        time.sleep(30)
    for s in (1,2,3):
        for method in ('unfiltered','filtered','aware','global','proposed'):
            path=source(s,method);r=load(path);summary=r['summary']
            if summary['completed'] or summary['contact']:continue
            if (summary.get('stop_reason') or '').startswith('Learning accuracy threshold not reached'):continue
            rerun_learning=summary['gate_time_s'] is None and not path.with_name(r['id']+'-policy-checkpoint.pkl').exists()
            profile=path.with_name(r['id']+'-final-profile.pkl.gz')
            if not profile.exists() and not rerun_learning:
                for attempt in range(3):
                    code=execute(['scripts/airflow/recover_recorded_profile.py',str(path),'--wall-limit','1800'],
                        Path('/tmp')/f'final-recover-s{s}-{method}.log',1950)
                    if code==0:break
                    if not path.with_name(r['id']+'-inference-replay.pkl.gz').exists():raise RuntimeError('Recovery failed without a checkpoint')
                else:raise RuntimeError('Bounded posterior reconstruction did not finish')
            destination=OUT/f'{method}-s{s}-completed'
            if destination.exists() and any(destination.iterdir()):raise RuntimeError('Existing continuation preserved; review its failure before retrying')
            args=continuation_args(s,method,path,destination)
            if rerun_learning:
                # Early runs did not save the stateful waypoint policy. A
                # partial reconstruction would change the method. Recompute
                # normally and verify the complete old prefix instead.
                i=args.index('--resume-monitoring-from');args[i]='--verify-execution-prefix'
                args[args.index('--wall-limit')+1]='6000'
            code=execute(args,Path('/tmp')/f'final-complete-s{s}-{method}.log',6150 if rerun_learning else 3750)
            if code:raise RuntimeError('Continuation failed; inspect its log')
            end=load(destination/'summary.json')[0]
            if not end['completed'] and not end['contact'] and not (end.get('stop_reason') or '').startswith('Learning accuracy threshold not reached'):
                raise RuntimeError('Continuation exhausted its declared budget')
    for script in ('package_final_comparison.py','plot_final_comparison.py','audit_final_comparison.py'):
        if execute(['scripts/airflow/'+script],Path('/tmp/final-finish-export.log'),600):raise RuntimeError('Final export or audit failed')
    print(json.dumps(dict(event='all_declared_trials_final',wall_s=time.monotonic()-began)),flush=True)

if __name__=='__main__':main()
