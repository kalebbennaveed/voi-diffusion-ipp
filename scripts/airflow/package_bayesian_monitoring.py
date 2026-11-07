"""Append audited task-directed learning experiments without replacing old data."""
import argparse,json,gzip,hashlib,shutil,zipfile,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def package(directory,site):
 audit=json.loads((directory/'audit.json').read_text());belief=json.loads((directory/'belief-audit.json').read_text())
 assert len(audit['checks'])==len(belief['checks'])==4
 out=site/'data/monitoring-bayesian-300';out.mkdir(parents=True,exist_ok=True)
 runs=[];warm=[];first=[];mbd=[];look=[];rejected=[]
 env=json.loads((directory/'environment.json').read_text())
 for scenario in (1,2):
  for method in ('aware','baseline'):
   ident=f's{scenario}-{method}';record=json.loads((directory/f'{ident}.json').read_text());protocol=json.loads((directory/f'{ident}-protocol.json').read_text());summary=record['summary'];a=np.asarray(record['samples']);rejected.extend(dict(id=ident,**call) for call in protocol.get('rejected_planning_calls',[]))
   assert summary['completed'] and summary['simulated_seconds']==300 and summary['contact'] is None
   visits=[r['label'] for r in env['regions'] if np.any(np.all((a[:,1:3]>=np.asarray(r['bounds'][0]))&(a[:,1:3]<=np.asarray(r['bounds'][1])),axis=1))]
   summary['regions_visited']=visits;summary['first_60s_mean_deficit']=float(np.interp(60,a[:,0],a[:,14])/60);summary['last_60s_mean_deficit']=float((a[-1,14]-np.interp(240,a[:,0],a[:,14]))/60)
   summary['selected_prefix_counts']={name:sum(p['selected_root']==i for p in record['plans']) for i,name in enumerate(('MBD','hold','alternative'))}
   for p in record['plans']:
    if p['timing']['compilation_included']:first.append(p['wall_seconds'])
    else:warm.append(p['wall_seconds']);mbd.append(p['timing']['optimization_validation']);look.append(p['timing']['lookahead']['total'])
   payload=json.dumps(record,separators=(',',':')).encode();encoded=gzip.compress(payload,mtime=0);assert gzip.decompress(encoded)==payload
   filename=ident+'.json.gz';(out/filename).write_bytes(encoded);shutil.copyfile(directory/f'{ident}-protocol.json',out/f'{ident}-protocol.json')
   runs.append(dict(id=ident,scenario=scenario,method=method,file=filename,decoded_sha256=hashlib.sha256(payload).hexdigest(),summary=summary))
 for f in ('environment.json','audit.json','belief-audit.json'):shutil.copyfile(directory/f,out/f)
 diagnostics=[]
 for file in ('exploration_benchmark.json','active_sensing_benchmark.json','active_sensing_w1v2_benchmark.json','thompson_benchmark.json','bayesian_budgeted_benchmark.json','bayesian_budgeted_w1v1_benchmark.json','bayesian_legacy_proposals_benchmark.json','bayesian_three_roots_benchmark.json','inducing_information_benchmark.json','monitoring_stall_diagnostic.json'):
  path=directory.parent/file
  if path.exists():diagnostics.append(dict(file=file,report=json.loads(path.read_text())))
 for folder in ('monitoring_active_sensing_300','monitoring_inducing_information_300','monitoring_bayesian_launch_diagnostic','monitoring_bayesian_two_branches_diagnostic','monitoring_bayesian_two_conditional_diagnostic'):
  path=directory.parent/folder/('s1-baseline.json' if folder=='monitoring_bayesian_two_conditional_diagnostic' else 's1-aware.json')
  if path.exists():
   r=json.loads(path.read_text());p=json.loads(path.with_name('s1-aware-protocol.json').read_text())
   diagnostics.append(dict(experiment=folder,summary=r['summary'],settings=p['legacy_settings'],warm_planning=[dict(t=x['t'],seconds=x['wall_seconds'],timing=x['timing']) for x in r['plans'] if not x['timing']['compilation_included']]))
 (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
 l=protocol['legacy_settings'];settings=protocol['settings']|dict(horizon_seconds=l['horizon'],commit_seconds=l['commit'])
 reached=sum('A' in r['summary']['regions_visited'] for r in runs)
 outcome=f"All four 300 s runs completed without contact. {reached} of four runs reached Region A. "
 outcome+=('The new method has not demonstrated the desired inter-region exploration; reducing uncertainty or passing audits does not establish that behavior.' if reached==0 else 'Compare region visits and monitoring cost below; one paired seed does not establish a general benefit.')
 for scenario in (1,2):
  pair={r['method']:r['summary']['integrated_deficit'] for r in runs if r['scenario']==scenario}
  change=100*(pair['aware']/pair['baseline']-1)
  outcome+=f" Scenario {scenario}: learning cost is {abs(change):.2f}% {'higher' if change>0 else 'lower'} than the matched comparator."
 manifest=dict(duration_s=300,state='complete',implementation='bayesian-mbd',planner_settings=settings,legacy_settings=l,initial_wind_knowledge='cold-start',execution_belief=protocol['execution_belief'],prior_configuration=protocol['prior_configuration'],method_labels=dict(aware='Task-directed learning',baseline='Current-belief planning'),method_descriptions=dict(aware='Future wind measurements can change the continuation decision. Task cost only.',baseline='Matched planner with a frozen forecast belief. Still learns from actual measurements.'),outcome=outcome,runtime=dict(warm_min_s=min(warm),warm_max_s=max(warm),warm_median_s=float(np.median(warm)),warm_calls=len(warm),mbd_median_s=float(np.median(mbd)),lookahead_median_s=float(np.median(look)),first_call_min_s=min(first),first_call_max_s=max(first),limit_s=1.,rejected_calls=rejected),runs=runs,verification='audit.json',source_available=True,method_document='method.html')
 assert max(warm)<=1
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 import html
 source=(ROOT/'docs/airflow/task_directed_learning.md').read_text()
 import re
 def inline(text):
  text=html.escape(text)
  text=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',text)
  return re.sub(r'\[([^]]+)\]\((https://[^)]+)\)',r'<a href="\2">\1</a>',text)
 blocks=[]
 for block in source.split('\n\n'):
  if block.startswith('# '):blocks.append('<h1>'+inline(block[2:])+'</h1>')
  elif block.startswith('## '):blocks.append('<h2>'+inline(block[3:])+'</h2>')
  elif block.startswith('- '):blocks.append('<ul>'+''.join('<li>'+inline(line[2:])+'</li>' for line in block.splitlines())+'</ul>')
  else:blocks.append('<p>'+inline(block)+'</p>')
 (out/'method.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Task-directed learning method</title><style>body{max-width:850px;margin:40px auto;padding:20px;color:#233e36;font:15px/1.75 system-ui}h1{font-size:30px;line-height:1.25}h2{font-size:21px;margin-top:30px}a{color:#176187}li{margin:12px 0}</style><a href="../../index.html#exploration-review">← Back to simulations</a>'+''.join(blocks))
 with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for path in (ROOT/'src').rglob('*.py'):z.write(path,str(path.relative_to(ROOT)))
  for name in ('run_bayesian_monitoring.py','exploration_monitoring.py','audit_bayesian_monitoring.py','package_bayesian_monitoring.py','legacy_monitoring.py','monitoring_mbd.py','monitoring_routes.py','run_monitoring_mbd.py','robot_review.py','audit_belief_comparison.py','benchmark_bayesian_budgeted.py','active_sensing_monitoring.py','run_active_sensing_monitoring.py','benchmark_active_sensing.py'):
   path=ROOT/'scripts/airflow'/name;z.write(path,str(path.relative_to(ROOT)))
  for name in ('test_bayesian_monitoring.py','test_active_sensing_monitoring.py'):
   path=ROOT/'tests'/name;z.write(path,str(path.relative_to(ROOT)))
  z.write(ROOT/'docs/airflow/task_directed_learning.md','METHOD.md');z.write(ROOT/'pyproject.toml','pyproject.toml')
  for path in out.glob('*.json'):z.write(path,'results/'+path.name)
  for r in runs:
   raw=np.load(directory/f"{r['id']}-execution.npz");buffer=io.BytesIO();np.savez_compressed(buffer,samples=raw['samples'],observation_rows=raw['observation_rows']);z.writestr(f"results/{r['id']}-controls-observations.npz",buffer.getvalue())
  for scenario in (1,2):
   for name in ('stationary-velocity.f32','metadata.json'):z.write(site/f'data/scenario{scenario}/{name}',f'scenario-site/data/scenario{scenario}/{name}')
  z.write(site/'app.js','scenario-site/app.js')
  z.writestr('REPRODUCE.txt','Install pyproject.toml dependencies. Set PYTHONPATH=src:scripts/airflow.\nFor S=1,2 and M=aware,baseline:\npython scripts/airflow/run_bayesian_monitoring.py --scenario S --method M --seconds 300 --levels 2 --proposals 14 --worlds 1 --validation 2 --validation-batch 4 --matched-prior --site scenario-site --output reproduced\npython scripts/airflow/audit_bayesian_monitoring.py --directory reproduced --site scenario-site\npython scripts/airflow/audit_belief_comparison.py --directory reproduced --site scenario-site\nWarm planning over 1 s stops the mission; use an external 240 s process cap. First-call compilation is separate. Full viewer records are served beside source.zip. Raw full plans and execution arrays are regenerated by these commands.\n')
 print(json.dumps(dict(directory=str(out),runtime=manifest['runtime'],outcome=outcome,archive_bytes=(out/'source.zip').stat().st_size)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--site',type=Path,required=True);a=p.parse_args();package(a.directory,a.site)
