"""Bounded declared run queue; never replaces existing experiment outputs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('queue',type=Path);a=p.parse_args();spec=json.loads(a.queue.read_text())
    began=time.monotonic();required=set(spec.get('required_ids',[]));waiting=Path(spec['wait_for'])
    while True:
        if waiting.exists():
            try:
                entries=json.loads(waiting.read_text())
                if required.issubset({e['id'] for e in entries}):break
            except (ValueError,KeyError):pass
        if time.monotonic()-began>7200:raise TimeoutError('Queue prerequisite did not finish within two hours')
        time.sleep(5)
    for job in spec['jobs']:
        destination=Path(job['args'][job['args'].index('--output')+1])
        if destination.exists() and any(destination.iterdir()):raise RuntimeError(f'Existing output preserved: {destination}')
        print(json.dumps(dict(event='job_start',name=job['name'])),flush=True)
        log=Path('/private/tmp')/('final-'+job['name']+'.log')
        limit=int(job['args'][job['args'].index('--wall-limit')+1])+150 if '--wall-limit' in job['args'] else 2550
        with log.open('w') as f:
            result=subprocess.run([sys.executable,'scripts/airflow/run_full_learning.py',*job['args']],stdout=f,stderr=subprocess.STDOUT,env=os.environ,timeout=limit)
        print(json.dumps(dict(event='job_end',name=job['name'],exit_code=result.returncode,log=str(log))),flush=True)
        if result.returncode:raise RuntimeError('Failed simulation: review log before continuing queue')

if __name__=='__main__':main()
