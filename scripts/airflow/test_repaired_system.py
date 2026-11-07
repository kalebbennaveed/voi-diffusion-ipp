"""Run every collected test file in isolation, retaining failures and timeouts."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/gp_safety_20261104/tests'

def run(path):
    start=time.perf_counter()
    name=str(path.relative_to(ROOT)).replace('/','__')
    log=OUT/(name+'.log');xml=OUT/(name+'.xml')
    with log.open('w') as f:
        try:
            result=subprocess.run([sys.executable,'-m','pytest','-q',str(path),f'--junitxml={xml}'],
                cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=120,
                env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            code=result.returncode
        except subprocess.TimeoutExpired:code='timeout'
    return dict(file=str(path.relative_to(ROOT)),exit_code=code,wall_seconds=time.perf_counter()-start,log=log.name)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    files=sorted((ROOT/'tests').rglob('test*.py'))
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(run,files):
            results.append(result)
            (OUT/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(result),flush=True)
