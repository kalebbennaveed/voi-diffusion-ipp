"""Run every collected test file in isolation, retaining failures and timeouts."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/gp_safety_20261104/continuation-retry'

def run(path):
    start=time.perf_counter()
    name=path.replace('/','__').replace(':','_')
    log=OUT/(name+'.log');xml=OUT/(name+'.xml')
    with log.open('w') as f:
        try:
            result=subprocess.run([sys.executable,'-m','pytest','-q',str(path),f'--junitxml={xml}'],
                cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=120,
                env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            code=result.returncode
        except subprocess.TimeoutExpired:code='timeout'
    return dict(file=path,exit_code=code,wall_seconds=time.perf_counter()-start,log=log.name)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    files=[line.strip() for line in Path('/tmp/gp-safety-test-collection.txt').read_text().splitlines() if line.startswith('tests/quarter/test_mission_continuation.py::')]
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for result in pool.map(run,files):
            results.append(result)
            (OUT/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(result),flush=True)
