"""A compact numerical report of the latest matched forward/reversed reruns."""
import argparse,hashlib,html,json
from pathlib import Path

METHODS=('proposed','blind','blind_filter')
LABELS={'forward':'Forward wind','reverse':'Reversed wind'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    integrity=json.loads((args.results/'execution_integrity.json').read_text())
    audited={r['file']:r for r in integrity['runs']};rows=[];records=[];cfg=None;hashes=None
    for case,label in LABELS.items():
        runs={}
        for method in METHODS:
            path=args.results/f'two_{case}_{method}_8101.json';d=json.loads(path.read_text())
            assert d['completed'] and d['mission_seconds']==300 and not d['plan_only'],path
            assert abs(audited[path.name]['reconstructed_cost']-d['cost'])<.0003,path
            if cfg is None:cfg=d['config'];hashes=d['source_hashes']
            assert d['config']==cfg and d['source_hashes']==hashes,path
            if d['success']:assert abs(d['completed_seconds']-300)<.011 and len(d['observations'])==300,path
            runs[method]=d
            records.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_hashes=d['source_hashes'],success=d['success'],seconds=d['completed_seconds'],recorded_cost=d['cost']))
        p=runs['proposed'];rows.append(dict(scenario=label,case=case,costs={m:runs[m]['cost'] if runs[m]['success'] else None for m in METHODS},successes={m:runs[m]['success'] for m in METHODS},reduction_percent={m:100*(1-p['cost']/runs[m]['cost']) if p['success'] and runs[m]['success'] else None for m in ('blind','blind_filter')}))
    report=dict(revision='Latest corrected forward/reversed rerun',mission_seconds=300,seed=8101,metric='Time integral of mean per-cell positive clarity deficit',units='deficit*s',lower_is_better=True,rows=rows,configuration=cfg,source_records=records,verification='Costs, motion and clearances reconstructed from the 100 Hz execution archives. Same settings across both cases and all methods. A terminated flight is excluded from comparisons.')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'latest-forward-reverse.json').write_text(json.dumps(report,indent=2))
    def cost(x):return 'Terminated' if x is None else f'{x:.2f}'
    def change(x):return 'Excluded' if x is None else f'{abs(x):.1f}% '+('less' if x>=0 else 'more')
    body='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Latest forward and reversed wind results</title><style>body{font:18px/1.5 system-ui;max-width:1100px;margin:60px auto;padding:0 24px;color:#203644}h1{font-size:29px;line-height:1.2}table{border-collapse:collapse;width:100%;margin:26px 0}td,th{padding:14px 10px;text-align:left;border-bottom:1px solid #ccd4dc}p{max-width:900px}a{color:#087967}.scroll{overflow:auto}</style></head><body><h1>Latest forward and reversed wind results</h1><p>Fresh five-minute missions using the latest corrections. Accumulated clarity deficit, in deficit·s. <b>Lower is better.</b></p><div class="scroll"><table><tr><th>Scenario</th><th>Proposed</th><th>Blind</th><th>Blind + filter</th><th>Proposed vs blind</th><th>Proposed vs filtered</th></tr>'''
    for r in rows:body+=f'<tr><td>{html.escape(r["scenario"])}</td>'+''.join(f'<td>{cost(r["costs"][m])}</td>' for m in METHODS)+''.join(f'<td>{change(r["reduction_percent"][m])}</td>' for m in ('blind','blind_filter'))+'</tr>'
    success=sum(r['success'] for r in records)
    body+=f'</table></div><p>{success}/6 flights completed the full 300 s without contact or other termination. Same vehicle, task, initial state, estimator and optimizer settings; seed 8101. This is one run per method and wind direction.</p><p><a href="latest-forward-reverse.json">Verified numerical records</a> · <a href="index.html#results">Detailed investigation</a></p></body></html>'
    (args.output/'latest-forward-reverse.html').write_text(body)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
