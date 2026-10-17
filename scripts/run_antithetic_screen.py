"""Execute the two predeclared antithetic development pairs, one worker at a time."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/wind_route_study/antithetic_protocol_01.json'
STATUS = ROOT / 'results/antithetic_screen_status_01.json'


def main():
    protocol = json.loads(PROTOCOL.read_text())
    if STATUS.exists():
        raise FileExistsError('Existing screen status: inspect it rather than duplicate work')
    status = {'state': 'running', 'jobs': []}
    env = os.environ | {'JAX_PLATFORMS': 'cpu'}
    for job in protocol['jobs']:
        for file, digest in protocol['frozen_sha256'].items():
            assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest, file
        row = {'name': job['name'], 'state': 'running', 'started': time.time()}
        status['jobs'].append(row)
        STATUS.write_text(json.dumps(status, indent=2)+'\n')
        print('START', job['name'], flush=True)
        output = ROOT / job['output']
        try:
            with (ROOT/'results'/f'{job["name"]}.log').open('x') as log:
                subprocess.run([sys.executable, 'scripts/run_budgeted_mission.py',
                                *job['arguments'], '--output', job['output']],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True)
                paths = [str(output/(m+'.npz')) for m in ['proposed', 'current_belief']]
                subprocess.run([sys.executable, 'scripts/audit_budgeted_mission.py', *paths],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True)
                subprocess.run([sys.executable, 'scripts/export_budgeted_demo.py', *paths],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True)
            row['state'] = 'complete'
        except Exception as exc:
            row['state'] = 'failed'
            row['error'] = repr(exc)
            status['state'] = 'failed'
            STATUS.write_text(json.dumps(status, indent=2)+'\n')
            raise
        row['ended'] = time.time()
        STATUS.write_text(json.dumps(status, indent=2)+'\n')
        print('COMPLETE', job['name'], flush=True)
    status['state'] = 'complete'
    STATUS.write_text(json.dumps(status, indent=2)+'\n')


if __name__ == '__main__':
    main()
