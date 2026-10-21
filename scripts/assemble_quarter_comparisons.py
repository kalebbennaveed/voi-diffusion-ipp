"""Assemble explicitly identified matched reruns, without editing recordings."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/european_quarter'

def main():
    out=BASE/'rrt_mbd_execution_300';out.mkdir(exist_ok=True)
    sources=[];configs={};records=[]
    for layout,case in [('two','forward'),('two','reverse'),('two','crosswind'),('two','commitment'),('multi','crosswind')]:
        source=BASE/('rrt_mbd_execution_four_site_300_v2' if layout=='multi' else 'rrt_mbd_execution_300_v1')
        group=[]
        for method in ('proposed','blind','blind_filter'):
            path=source/f'{layout}_{case}_{method}_8101.json';d=json.loads(path.read_text())
            assert d['completed'] and d['success'] and d['completed_seconds']>=299.99, f'Unresolved flight: {path}'
            group.append(d)
            paths={path.name}
            for plan in d['plans']:
                paths.add(plan['execution_audit']);paths.update(plan.get('chain_audits',[]))
                if plan.get('audit'):paths.add(plan['audit'])
            for name in paths:shutil.copy2(source/name,out/name)
            sources.append(dict(file=path.name,source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            records.append(d)
        assert all(d['config']==group[0]['config'] for d in group)
        configs[f'{layout}/{case}']=group[0]['config']
    # The unfiltered four-site controller does not depend on the margin. Its
    # independent rerun must reproduce every measured frame and observation.
    old=json.loads((BASE/'rrt_mbd_execution_300_v1/multi_crosswind_blind_8101.json').read_text())
    new=next(d for d in records if d['layout']=='multi' and d['method']=='blind')
    assert old['frames']==new['frames'] and old['observations']==new['observations']
    experiment=dict(description='All five environments, all three methods, fresh 300 s runs. Four-site execution clearance tuned after a recorded contact; all three methods rerun with the common setting.',
        configurations=configs,recordings=sources,
        previous_attempt='output/european_quarter/rrt_mbd_execution_300_v1/multi_crosswind_blind_filter_8101.json',
        four_site_unfiltered_repeat_identical=True,
        timing_note='Some environment reruns overlapped. Per-run timings are measured wall times, not isolated hardware benchmarks.')
    (out/'experiment.json').write_text(json.dumps(experiment,indent=2))
    print('Assembled 15 verified complete recordings, unchanged from their source files:',out)

if __name__=='__main__':main()
